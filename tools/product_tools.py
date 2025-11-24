# tools/products_tools.py
"""
Clean, minimal product tools for a Sales Agent.

Expect CSV at: <this_file_dir>/data/product.csv

Design:
- Internal functions return structured dicts (machine-friendly).
- Thin @tool wrappers returned human-friendly strings for use with LangChain.
- Minimal error handling: only catch and report expected file/IO errors.
"""

import os
import tempfile
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
from langchain.tools import tool

# --- Configuration ---
PRODUCTS_CSV = os.path.join(os.path.dirname(__file__), "data", "my-fastapi-project\tools\data\product.csv")
NUM_RECS_DEFAULT = 5

# --- Helpers ---


def _load_products_df(csv_path: str = PRODUCTS_CSV) -> pd.DataFrame:
    """
    Load CSV into a DataFrame and ensure numeric columns have correct dtypes.

    Raises:
        FileNotFoundError: if CSV doesn't exist.
        ValueError: if required columns missing or cast fails.
    """
    df = pd.read_csv(csv_path)
    required_cols = {"product_id", "product_name", "product_description", "type", "price", "rating", "inventory_count"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Cast numeric columns and normalize string columns
    df["price"] = pd.to_numeric(df["price"], errors="raise")
    df["rating"] = pd.to_numeric(df["rating"], errors="raise")
    df["inventory_count"] = pd.to_numeric(df["inventory_count"], errors="raise").astype(int)
    df["product_id"] = df["product_id"].astype(str)
    df["product_name"] = df["product_name"].astype(str)
    df["product_description"] = df["product_description"].astype(str)
    df["type"] = df["type"].astype(str)
    return df


def _save_products_df(df: pd.DataFrame, csv_path: str = PRODUCTS_CSV) -> None:
    """
    Persist DataFrame back to CSV atomically (write to temp then replace).
    """
    dirpath = os.path.dirname(csv_path)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dirpath, newline='', encoding="utf-8") as tf:
        df.to_csv(tf.name, index=False)
        tmpname = tf.name
    # replace original file
    os.replace(tmpname, csv_path)


def _format_product_row(row: pd.Series) -> str:
    return f"{row['product_id']} — {row['product_name']} (${row['price']:.2f}, rating {row['rating']}, type {row['type']})"


# --- Core (machine-friendly) functions ---


def search_product_by_name_internal(product_name: str, csv_path: str = PRODUCTS_CSV) -> Dict[str, Any]:
    """
    Search products by keyword in name or description.
    Returns dict:
      {"success": True, "matches": [ {product_row_dict}, ... ] } or
      {"success": False, "error": "..."}
    """
    df = _load_products_df(csv_path)
    q = product_name.strip().lower()
    if not q:
        return {"success": False, "error": "Empty search query."}

    mask = (
        df["product_name"].str.lower().str.contains(q, na=False)
        | df["product_description"].str.lower().str.contains(q, na=False)
    )
    matches = df[mask]
    results = matches.to_dict(orient="records")
    return {"success": True, "count": len(results), "matches": results}


def filter_products_internal(
    product_type: Optional[str] = None,
    min_rating: Optional[float] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    top_n: int = NUM_RECS_DEFAULT,
    csv_path: str = PRODUCTS_CSV,
) -> Dict[str, Any]:
    """
    Filter products and return top_n recommendations sorted by rating (desc) then price (asc).
    Returns dict with success flag and list of product dicts.
    """
    df = _load_products_df(csv_path)
    filtered = df

    if product_type:
        filtered = filtered[filtered["type"].str.lower() == product_type.strip().lower()]

    if min_rating is not None:
        filtered = filtered[filtered["rating"] >= float(min_rating)]

    if min_price is not None:
        filtered = filtered[filtered["price"] >= float(min_price)]

    if max_price is not None:
        filtered = filtered[filtered["price"] <= float(max_price)]

    if filtered.empty:
        return {"success": True, "count": 0, "recommendations": []}

    # Sort by rating desc, price asc
    sorted_df = filtered.sort_values(by=["rating", "price"], ascending=[False, True])
    recs = sorted_df.head(int(top_n)).to_dict(orient="records")
    return {"success": True, "count": len(recs), "recommendations": recs}


def check_inventory_internal(product_id: str, csv_path: str = PRODUCTS_CSV) -> Dict[str, Any]:
    """
    Returns stock information for product_id.
    """
    df = _load_products_df(csv_path)
    mask = df["product_id"].str.upper() == str(product_id).strip().upper()
    matches = df[mask]
    if matches.empty:
        return {"success": False, "error": "not_found", "product_id": product_id}
    row = matches.iloc[0]
    return {
        "success": True,
        "product_id": row["product_id"],
        "product_name": row["product_name"],
        "price": float(row["price"]),
        "rating": float(row["rating"]),
        "inventory_count": int(row["inventory_count"]),
        "in_stock": int(row["inventory_count"]) > 0,
    }


def checkout_internal(product_id: str, quantity: int = 1, csv_path: str = PRODUCTS_CSV) -> Dict[str, Any]:
    """
    Reduce inventory by `quantity` and persist. Returns dict:
      {"success": True, "order": {...}} or {"success": False, "error": "..."}
    Note: This is a simple CSV-backed checkout mechanism (not safe for concurrent writes in production).
    """
    if quantity <= 0:
        return {"success": False, "error": "invalid_quantity", "message": "Quantity must be >= 1"}

    df = _load_products_df(csv_path)
    mask = df["product_id"].str.upper() == str(product_id).strip().upper()
    idx = df.index[mask]
    if len(idx) == 0:
        return {"success": False, "error": "not_found", "product_id": product_id}

    i = idx[0]
    available = int(df.at[i, "inventory_count"])
    if available < quantity:
        return {"success": False, "error": "insufficient_inventory", "available": available}

    # perform update and save
    df.at[i, "inventory_count"] = available - quantity
    _save_products_df(df, csv_path)

    order = {
        "order_id": f"ORD-{os.urandom(4).hex()}",
        "product_id": df.at[i, "product_id"],
        "product_name": df.at[i, "product_name"],
        "qty": int(quantity),
        "unit_price": float(df.at[i, "price"]),
        "total_price": round(float(df.at[i, "price"]) * int(quantity), 2),
    }
    return {"success": True, "order": order}


# --- Tool wrappers (human-friendly strings for LangChain) ---


def _dict_to_text_search(result: Dict[str, Any]) -> str:
    if not result.get("success", False):
        return f"Error: {result.get('error', 'unknown')}. {result.get('message','')}"
    if result["count"] == 0:
        return "No products found."
    lines = [f"Products found ({result['count']}):"]
    for r in result["matches"]:
        lines.append(f"• {_format_product_row(pd.Series(r))}")
    return "\n".join(lines)


def _dict_to_text_filter(result: Dict[str, Any]) -> str:
    if not result.get("success", False):
        return f"Error: {result.get('error', 'unknown')}"
    if result["count"] == 0:
        return "No products match the given filters."
    lines = [f"Top {result['count']} recommendation(s):"]
    for r in result["recommendations"]:
        lines.append(f"• {_format_product_row(pd.Series(r))} — inventory: {r.get('inventory_count', 'N/A')}")
    return "\n".join(lines)


def _dict_to_text_inventory(result: Dict[str, Any]) -> str:
    if not result.get("success", False):
        return f"Product {result.get('product_id')} not found."
    return (
        f"{result['product_name']} — {'in stock' if result['in_stock'] else 'out of stock'} "
        f"({result['inventory_count']} units). Price: ${result['price']:.2f}, Rating: {result['rating']}."
    )


def _dict_to_text_checkout(result: Dict[str, Any]) -> str:
    if not result.get("success", False):
        err = result.get("error", "unknown")
        if err == "insufficient_inventory":
            return f"Checkout failed: only {result.get('available', 0)} units available."
        if err == "not_found":
            return f"Checkout failed: product {result.get('product_id')} not found."
        return f"Checkout failed: {err}"
    o = result["order"]
    return (
        f"✅ Checkout successful: {o['qty']} × {o['product_name']} (Order ID: {o['order_id']}). "
        f"Total: ${o['total_price']:.2f}"
    )


@tool
def search_product_by_name(product_name: str) -> str:
    """Tool wrapper for product search (human-friendly string output)."""
    try:
        result = search_product_by_name_internal(product_name)
    except FileNotFoundError:
        return f"Product database not found at {PRODUCTS_CSV}."
    except Exception as exc:
        # Unexpected error — let this be visible to developer instead of swallowing
        return f"Error searching product: {exc}"
    return _dict_to_text_search(result)


@tool
def filter_products(
    product_type: Optional[str] = None,
    min_rating: Optional[float] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    top_n: int = NUM_RECS_DEFAULT,
) -> str:
    """Tool wrapper for filtering products (returns formatted top recommendations)."""
    try:
        result = filter_products_internal(
            product_type=product_type,
            min_rating=min_rating,
            min_price=min_price,
            max_price=max_price,
            top_n=top_n,
        )
    except FileNotFoundError:
        return f"Product database not found at {PRODUCTS_CSV}."
    except Exception as exc:
        return f"Error filtering products: {exc}"
    return _dict_to_text_filter(result)


@tool
def check_inventory(product_id: str) -> str:
    """Tool wrapper for inventory check."""
    try:
        result = check_inventory_internal(product_id)
    except FileNotFoundError:
        return f"Product database not found at {PRODUCTS_CSV}."
    except Exception as exc:
        return f"Error checking inventory: {exc}"
    return _dict_to_text_inventory(result)


@tool
def checkout_product(product_id: str, quantity: int = 1) -> str:
    """Tool wrapper for checkout (will persist inventory change)."""
    try:
        result = checkout_internal(product_id, int(quantity))
    except FileNotFoundError:
        return f"Product database not found at {PRODUCTS_CSV}."
    except Exception as exc:
        return f"Error during checkout: {exc}"
    return _dict_to_text_checkout(result)


# Export a list for convenience (if your loader expects it)
tools = [search_product_by_name, filter_products, check_inventory, checkout_product]
