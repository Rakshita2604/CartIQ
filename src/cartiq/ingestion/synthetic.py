"""Deterministic, hand-curated Indian grocery catalog for tests and local development.

No randomness: the same code always yields the same rows. Brands deliberately recur across
categories (Amul, Tata, Fortune, Maggi) and near-duplicates exist (toned vs full cream milk)
so that later phases have realistic ambiguity to resolve.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from cartiq.ingestion.raw import RawCatalogRow

SYNTHETIC_CATALOG_VERSION = "synthetic-v1"


@dataclass(frozen=True, slots=True)
class _Line:
    category: str
    sub_category: str
    product_type: str
    brand: str | None
    name: str
    sizes: Sequence[tuple[str, int]]  # (size label, sale price in rupees)


_DAIRY = ("Bakery, Cakes & Dairy", "Dairy")
_STAPLES = ("Foodgrains, Oil & Masala", "Atta, Flours & Sooji")
_RICE = ("Foodgrains, Oil & Masala", "Rice & Rice Products")
_DALS = ("Foodgrains, Oil & Masala", "Dals & Pulses")
_OIL = ("Foodgrains, Oil & Masala", "Edible Oils & Ghee")
_MASALA = ("Foodgrains, Oil & Masala", "Masalas & Spices")
_SALT_SUGAR = ("Foodgrains, Oil & Masala", "Salt, Sugar & Jaggery")
_VEG = ("Fruits & Vegetables", "Fresh Vegetables")
_FRUIT = ("Fruits & Vegetables", "Fresh Fruits")
_BEVERAGES = ("Beverages", "Tea")
_COFFEE = ("Beverages", "Coffee")
_SOFT = ("Beverages", "Energy & Soft Drinks")
_JUICE = ("Beverages", "Fruit Juices & Drinks")
_SNACKS = ("Snacks & Branded Foods", "Chips & Namkeen")
_BISCUITS = ("Snacks & Branded Foods", "Biscuits & Cookies")
_NOODLES = ("Snacks & Branded Foods", "Noodle, Pasta, Vermicelli")
_SAUCES = ("Snacks & Branded Foods", "Spreads, Sauces, Ketchup")
_BREAD = ("Bakery, Cakes & Dairy", "Breads & Buns")
_EGGS = ("Eggs, Meat & Fish", "Eggs")
_LAUNDRY = ("Cleaning & Household", "Detergents & Dishwash")
_ORAL = ("Beauty & Hygiene", "Oral Care")
_BATH = ("Beauty & Hygiene", "Bath & Hand Wash")
_HAIR = ("Beauty & Hygiene", "Hair Care")


def _line(
    group: tuple[str, str], product_type: str, brand: str | None, name: str, *sizes: tuple[str, int]
) -> _Line:
    return _Line(group[0], group[1], product_type, brand, name, sizes)


_LINES: tuple[_Line, ...] = (
    # Dairy
    _line(_DAIRY, "Milk", "Amul", "Taaza Toned Milk", ("500 ml", 28), ("1 l", 54)),
    _line(_DAIRY, "Milk", "Amul", "Gold Full Cream Milk", ("500 ml", 34), ("1 l", 66)),
    _line(_DAIRY, "Milk", "Mother Dairy", "Toned Milk", ("500 ml", 27), ("1 l", 54)),
    _line(_DAIRY, "Milk", "Mother Dairy", "Cow Milk", ("500 ml", 29)),
    _line(_DAIRY, "Milk", "Nandini", "Pasteurised Toned Milk", ("500 ml", 24), ("1 l", 46)),
    _line(_DAIRY, "Curd", "Amul", "Masti Dahi", ("400 g", 35), ("1 kg", 75)),
    _line(_DAIRY, "Curd", "Mother Dairy", "Classic Curd", ("400 g", 35)),
    _line(_DAIRY, "Curd", "Nestle", "a+ Nourish Dahi", ("400 g", 45)),
    _line(_DAIRY, "Paneer", "Amul", "Malai Paneer", ("200 g", 90), ("500 g", 220)),
    _line(_DAIRY, "Paneer", "Mother Dairy", "Fresh Paneer", ("200 g", 85)),
    _line(_DAIRY, "Butter", "Amul", "Pasteurised Butter", ("100 g", 58), ("500 g", 275)),
    _line(_DAIRY, "Butter", "Amul", "Garlic & Herbs Butter", ("100 g", 65)),
    _line(_DAIRY, "Butter", "Britannia", "Salted Butter", ("100 g", 56)),
    _line(_DAIRY, "Cheese", "Amul", "Cheese Slices", ("200 g", 140)),
    _line(_DAIRY, "Cheese", "Britannia", "Cheese Cubes", ("200 g", 135)),
    # Staples
    _line(_STAPLES, "Atta", "Aashirvaad", "Shudh Chakki Atta", ("5 kg", 245), ("10 kg", 470)),
    _line(_STAPLES, "Atta", "Pillsbury", "Chakki Fresh Atta", ("5 kg", 230)),
    _line(_STAPLES, "Atta", "Fortune", "Chakki Fresh Atta", ("5 kg", 225), ("10 kg", 440)),
    _line(_STAPLES, "Sooji", "Rajdhani", "Sooji Rava", ("500 g", 32), ("1 kg", 60)),
    _line(_STAPLES, "Besan", "Rajdhani", "Besan", ("500 g", 60)),
    _line(
        _RICE, "Basmati Rice", "India Gate", "Basmati Rice Classic", ("1 kg", 190), ("5 kg", 900)
    ),
    _line(
        _RICE, "Basmati Rice", "Daawat", "Rozana Super Basmati Rice", ("1 kg", 105), ("5 kg", 495)
    ),
    _line(_RICE, "Basmati Rice", "Fortune", "Everyday Basmati Rice", ("1 kg", 95)),
    _line(_RICE, "Raw Rice", "Organic Tattva", "Sona Masoori Rice", ("1 kg", 110)),
    _line(_RICE, "Poha", "Tata Sampann", "Thick Poha", ("500 g", 45)),
    _line(_DALS, "Toor Dal", "Tata Sampann", "Unpolished Toor Dal", ("500 g", 95), ("1 kg", 180)),
    _line(_DALS, "Toor Dal", "Fortune", "Arhar Dal", ("1 kg", 165)),
    _line(_DALS, "Moong Dal", "Tata Sampann", "Unpolished Moong Dal", ("500 g", 80), ("1 kg", 155)),
    _line(_DALS, "Chana Dal", "Rajdhani", "Chana Dal", ("1 kg", 110)),
    _line(_DALS, "Masoor Dal", "Tata Sampann", "Masoor Dal Whole", ("500 g", 70)),
    _line(_DALS, "Rajma", "Rajdhani", "Chitra Rajma", ("500 g", 95)),
    _line(
        _OIL,
        "Sunflower Oil",
        "Fortune",
        "Sunlite Refined Sunflower Oil",
        ("1 l", 150),
        ("5 l", 720),
    ),
    _line(_OIL, "Sunflower Oil", "Saffola", "Gold Refined Cooking Oil", ("1 l", 190), ("5 l", 900)),
    _line(_OIL, "Mustard Oil", "Fortune", "Kachi Ghani Mustard Oil", ("1 l", 170)),
    _line(_OIL, "Mustard Oil", "Patanjali", "Kachi Ghani Mustard Oil", ("1 l", 165)),
    _line(_OIL, "Ghee", "Amul", "Pure Ghee", ("500 ml", 315), ("1 l", 620)),
    _line(_OIL, "Ghee", "Patanjali", "Cow Ghee", ("500 ml", 330), ("1 l", 650)),
    _line(_MASALA, "Powdered Spices", "MDH", "Deggi Mirch", ("100 g", 85)),
    _line(_MASALA, "Powdered Spices", "Everest", "Turmeric Powder", ("100 g", 38), ("200 g", 72)),
    _line(_MASALA, "Blended Masalas", "MDH", "Garam Masala", ("100 g", 92)),
    _line(_MASALA, "Blended Masalas", "Everest", "Chicken Masala", ("100 g", 80)),
    _line(_MASALA, "Whole Spices", "Catch", "Jeera Whole", ("100 g", 75)),
    _line(_SALT_SUGAR, "Salt", "Tata", "Salt Iodised", ("1 kg", 28)),
    _line(_SALT_SUGAR, "Salt", "Aashirvaad", "Iodised Salt", ("1 kg", 25)),
    _line(_SALT_SUGAR, "Sugar", "Madhur", "Pure Sugar", ("1 kg", 55), ("5 kg", 265)),
    _line(_SALT_SUGAR, "Jaggery", "Organic Tattva", "Jaggery Powder", ("500 g", 90)),
    # Fruits & vegetables (loose produce has no brand)
    _line(_VEG, "Onion", None, "Fresh Onion", ("1 kg", 40), ("2 kg", 78)),
    _line(_VEG, "Potato", None, "Fresh Potato", ("1 kg", 35), ("2 kg", 68)),
    _line(_VEG, "Tomato", None, "Fresh Tomato Hybrid", ("500 g", 24), ("1 kg", 45)),
    _line(_VEG, "Herbs & Seasonings", None, "Fresh Coriander Leaves", ("100 g", 15)),
    _line(_VEG, "Herbs & Seasonings", None, "Fresh Green Chilli", ("100 g", 12)),
    _line(_VEG, "Ginger & Garlic", None, "Fresh Ginger", ("200 g", 30)),
    _line(_VEG, "Ginger & Garlic", None, "Fresh Garlic", ("200 g", 45)),
    _line(_VEG, "Leafy Vegetables", None, "Fresh Palak Spinach", ("250 g", 25)),
    _line(_FRUIT, "Banana", None, "Robusta Banana", ("6 pcs", 42), ("12 pcs", 80)),
    _line(_FRUIT, "Apple", None, "Shimla Apple", ("4 pcs", 120)),
    _line(_FRUIT, "Citrus", None, "Fresh Lemon", ("250 g", 30)),
    # Beverages
    _line(
        _BEVERAGES, "Leaf & Dust Tea", "Tata Tea", "Premium Leaf Tea", ("250 g", 140), ("1 kg", 530)
    ),
    _line(_BEVERAGES, "Leaf & Dust Tea", "Red Label", "Natural Care Tea", ("250 g", 150)),
    _line(_BEVERAGES, "Green Tea", "Tetley", "Green Tea Lemon & Honey", ("25 pcs", 160)),
    _line(
        _COFFEE,
        "Instant Coffee",
        "Nescafe",
        "Classic Instant Coffee",
        ("50 g", 190),
        ("100 g", 360),
    ),
    _line(_COFFEE, "Instant Coffee", "Bru", "Instant Coffee", ("100 g", 245)),
    _line(_SOFT, "Cold Drinks", "Coca-Cola", "Soft Drink", ("750 ml", 40), ("6 x 300 ml", 210)),
    _line(_SOFT, "Cold Drinks", "Thums Up", "Soft Drink", ("750 ml", 40)),
    _line(_SOFT, "Cold Drinks", "Sprite", "Lime Flavoured Soft Drink", ("750 ml", 40)),
    _line(_JUICE, "Juices", "Real", "Mixed Fruit Juice", ("1 l", 125)),
    _line(_JUICE, "Juices", "Maaza", "Mango Drink", ("600 ml", 40), ("6 x 200 ml", 120)),
    _line(_JUICE, "Juices", "Frooti", "Mango Drink", ("6 x 200 ml", 110)),
    # Snacks & branded foods
    _line(_SNACKS, "Chips", "Lay's", "India's Magic Masala Chips", ("52 g", 20), ("90 g", 40)),
    _line(_SNACKS, "Chips", "Lay's", "Classic Salted Chips", ("52 g", 20)),
    _line(_SNACKS, "Namkeen", "Haldiram's", "Aloo Bhujia", ("200 g", 55), ("400 g", 105)),
    _line(_SNACKS, "Namkeen", "Haldiram's", "Moong Dal", ("200 g", 55)),
    _line(_SNACKS, "Namkeen", "Bikaji", "Bikaneri Bhujia", ("400 g", 100)),
    _line(
        _BISCUITS,
        "Glucose Biscuits",
        "Parle",
        "Parle-G Gold Biscuits",
        ("100 g", 10),
        ("1 kg", 110),
    ),
    _line(_BISCUITS, "Cream Biscuits", "Britannia", "Bourbon Cream Biscuits", ("120 g", 30)),
    _line(_BISCUITS, "Marie Biscuits", "Britannia", "Marie Gold Biscuits", ("250 g", 40)),
    _line(_BISCUITS, "Cookies", "Sunfeast", "Dark Fantasy Choco Fills", ("75 g", 40)),
    _line(
        _NOODLES,
        "Instant Noodles",
        "Maggi",
        "2-Minute Masala Noodles",
        ("70 g", 14),
        ("12 x 70 g", 168),
    ),
    _line(_NOODLES, "Instant Noodles", "Yippee", "Magic Masala Noodles", ("4 x 70 g", 56)),
    _line(_NOODLES, "Pasta", "Del Monte", "Penne Pasta", ("500 g", 110)),
    _line(_NOODLES, "Vermicelli", "Bambino", "Roasted Vermicelli", ("400 g", 45)),
    _line(_SAUCES, "Ketchup", "Maggi", "Rich Tomato Ketchup", ("500 g", 115), ("1 kg", 160)),
    _line(_SAUCES, "Ketchup", "Kissan", "Fresh Tomato Ketchup", ("850 g", 140)),
    _line(_SAUCES, "Jam", "Kissan", "Mixed Fruit Jam", ("500 g", 160)),
    _line(_BREAD, "Bread", "Britannia", "100% Whole Wheat Bread", ("400 g", 55)),
    _line(_BREAD, "Bread", "Modern", "Sandwich White Bread", ("400 g", 45)),
    _line(_BREAD, "Pav", "Britannia", "Pav", ("200 g", 30)),
    _line(_EGGS, "Farm Eggs", None, "Farm Fresh White Eggs", ("6 pcs", 48), ("12 pcs", 90)),
    _line(_EGGS, "Farm Eggs", "Eggoz", "Nutrition Brown Eggs", ("6 pcs", 75)),
    # Household & personal care
    _line(
        _LAUNDRY,
        "Detergent Powder",
        "Surf Excel",
        "Easy Wash Detergent Powder",
        ("1 kg", 140),
        ("4 kg", 520),
    ),
    _line(
        _LAUNDRY, "Detergent Powder", "Tide", "Plus Jasmine & Rose Detergent Powder", ("1 kg", 120)
    ),
    _line(_LAUNDRY, "Liquid Detergent", "Surf Excel", "Matic Front Load Liquid", ("1 l", 230)),
    _line(_LAUNDRY, "Dishwash Bars", "Vim", "Dishwash Bar", ("200 g", 20), ("3 x 200 g", 55)),
    _line(_LAUNDRY, "Dishwash Liquid", "Vim", "Lemon Dishwash Liquid Gel", ("500 ml", 115)),
    _line(_ORAL, "Toothpaste", "Colgate", "Strong Teeth Toothpaste", ("200 g", 110)),
    _line(_ORAL, "Toothpaste", "Pepsodent", "Germicheck Toothpaste", ("150 g", 85)),
    _line(_ORAL, "Toothpaste", "Patanjali", "Dant Kanti Toothpaste", ("200 g", 95)),
    _line(_BATH, "Bathing Bars", "Lifebuoy", "Total 10 Soap", ("4 x 125 g", 140)),
    _line(_BATH, "Bathing Bars", "Dettol", "Original Soap", ("3 x 125 g", 130)),
    _line(_BATH, "Hand Wash", "Dettol", "Original Liquid Hand Wash", ("200 ml", 99)),
    _line(_HAIR, "Shampoo", "Clinic Plus", "Strong & Long Shampoo", ("340 ml", 210)),
    _line(_HAIR, "Shampoo", "Head & Shoulders", "Anti Dandruff Shampoo", ("340 ml", 380)),
    _line(
        _HAIR, "Hair Oil", "Parachute", "100% Pure Coconut Oil", ("250 ml", 120), ("500 ml", 230)
    ),
)


def synthetic_catalog_rows() -> list[RawCatalogRow]:
    rows: list[RawCatalogRow] = []
    for line in _LINES:
        for size, rupees in line.sizes:
            index = len(rows) + 1
            sale = Decimal(rupees)
            # Every third SKU is discounted 10% off MRP; the rest sell at MRP.
            mrp = (sale * Decimal("1.10")).quantize(Decimal(1)) if index % 3 == 0 else sale
            name = " ".join(part for part in (line.brand, line.name, size) if part)
            rows.append(
                RawCatalogRow(
                    source_id=f"syn-{index:04d}",
                    name=name,
                    category=line.category,
                    sub_category=line.sub_category,
                    product_type=line.product_type,
                    brand=line.brand or "",
                    sale_price=str(sale),
                    market_price=str(mrp),
                    description=f"{line.name} ({line.product_type}).",
                )
            )
    return rows
