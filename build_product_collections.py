"""
Build the ChromaDB embedding collections for the Product Category Classifier.

Replaces the two collections in `emb_collection/`:
  - intent_meaning_collection: one embedded description ("meaning") per category
  - sample_avg_embeddings_collection: average embedding of synthetic product
    examples per category

Categories: shoes, groceries, shirts, furniture

Run once from the repo root:
    python build_product_collections.py
"""

import numpy as np
import pandas as pd
from google import genai
from chromadb import PersistentClient

from src.utils import add_gemini_embeddings

PROJECT = "deron-innovations"
LOCATION = "us-central1"
CHROMA_PATH = "emb_collection"
EMBEDDING_MODEL = "gemini-embedding-001"

CATEGORY_MEANINGS = {
    "shoes": (
        "Footwear products worn on the feet. This includes sneakers, running shoes, "
        "trainers, boots, sandals, flip flops, loafers, heels, dress shoes, slippers, "
        "cleats and any other kind of shoe for men, women or children. Typical attributes "
        "include shoe size, sole material, lace or slip-on style, and use cases such as "
        "running, hiking, work safety or formal wear."
    ),
    "groceries": (
        "Food, drink and everyday consumable household items typically bought at a "
        "supermarket. This includes fresh produce like fruit and vegetables, dairy such as "
        "milk and cheese, bread and bakery items, meat and fish, pantry staples like rice, "
        "pasta, flour and canned goods, snacks, breakfast cereals, beverages such as juice, "
        "coffee and tea, frozen food, and condiments and spices. Typical attributes include "
        "weight or volume, organic or free-range labels, and pack sizes."
    ),
    "shirts": (
        "Upper-body clothing garments. This includes t-shirts, polo shirts, dress shirts, "
        "button-downs, blouses, flannel shirts, henleys, long-sleeve and short-sleeve tops "
        "for men, women or children. Typical attributes include fabric such as cotton, "
        "linen or polyester, fit such as slim or regular, collar style, sleeve length, "
        "size from XS to XXL, and patterns like plain, striped or checked."
    ),
    "furniture": (
        "Large functional items used to furnish homes and offices. This includes sofas, "
        "couches, armchairs, dining tables, coffee tables, desks, office chairs, beds, "
        "mattresses, bed frames, wardrobes, dressers, bookshelves, cabinets, TV stands and "
        "outdoor patio sets. Typical attributes include material such as oak, walnut, metal "
        "or upholstered fabric, dimensions, seating capacity, and assembly requirements."
    ),
}

CATEGORY_EXAMPLES = {
    "shoes": [
        "Men's leather running sneakers size 10",
        "Women's white canvas low-top trainers",
        "Kids' velcro strap school shoes black",
        "Waterproof hiking boots with ankle support",
        "Classic suede desert boots tan",
        "Ladies' block heel ankle boots",
        "Steel toe cap safety work boots",
        "Slip-on memory foam loafers brown",
        "Breathable mesh gym trainers grey",
        "Leather brogue dress shoes oxford style",
        "Summer beach flip flops rubber sole",
        "Women's strappy gladiator sandals",
        "Fleece-lined winter slippers with hard sole",
        "Football boots with moulded studs",
        "Trail running shoes with grippy outsole",
        "High-top basketball sneakers red and white",
        "Ballet flats with cushioned insole",
        "Chelsea boots black leather elastic side panel",
        "Toddler first-walker soft sole shoes",
        "Wide-fit orthopedic walking shoes",
        "Patent leather stiletto heels 4 inch",
        "Canvas espadrilles with jute sole",
        "Waterproof wellington rain boots green",
        "Cycling shoes with cleat compatibility",
        "Men's boat shoes navy leather",
        "Retro skate shoes with padded tongue",
        "Lightweight marathon racing flats",
        "Sheepskin-lined snow boots",
        "Formal monk strap shoes dark brown",
        "Cross-training shoes with lateral support",
        "Platform sandals with cork footbed",
        "Golf shoes with soft spikes white",
        "Slip-resistant kitchen work shoes",
        "Moccasin house shoes hand stitched",
        "Kids light-up trainers with LED soles",
        "Vegan leather ankle boots block heel",
        "Tennis court shoes non-marking sole",
        "Climbing shoes with downturned toe",
        "Cushioned road running shoes neutral gait",
        "Fur-trimmed fashion winter boots",
    ],
    "groceries": [
        "Organic whole milk 2 litre bottle",
        "Free-range large eggs box of 12",
        "Wholemeal sliced bread loaf 800g",
        "Bananas bunch of 6 fair trade",
        "Cheddar cheese mature block 400g",
        "Basmati rice 5kg bag",
        "Extra virgin olive oil 750ml",
        "Chicken breast fillets 1kg pack",
        "Fresh Atlantic salmon fillets 2 pieces",
        "Penne pasta 500g durum wheat",
        "Chopped tomatoes canned 400g 4 pack",
        "Greek natural yogurt 500g pot",
        "Ground arabica coffee 227g bag",
        "English breakfast tea 80 bags",
        "Orange juice freshly squeezed 1 litre",
        "Frozen garden peas 1kg bag",
        "Crunchy peanut butter 340g jar",
        "Strawberry jam 454g jar",
        "Salted butter 250g block",
        "Baby spinach leaves washed 200g",
        "Red seedless grapes 500g punnet",
        "Porridge oats rolled 1kg",
        "Dark chocolate bar 70 percent cocoa 100g",
        "Sea salt and black pepper grinder set",
        "Sparkling mineral water 6 x 1 litre",
        "Granulated white sugar 1kg bag",
        "Plain flour 1.5kg for baking",
        "Tortilla wraps 8 pack white",
        "Beef mince 5 percent fat 500g",
        "Honey squeezable bottle 340g",
        "Salted crisps multipack 6 bags",
        "Corn flakes breakfast cereal 500g box",
        "Fresh broccoli head loose",
        "Avocados ripe and ready 2 pack",
        "Tomato ketchup 460ml squeezy bottle",
        "Long grain brown rice 1kg",
        "Almond milk unsweetened 1 litre carton",
        "Frozen margherita pizza thin crust",
        "Canned chickpeas in water 400g",
        "Mixed nuts and raisins snack pack 200g",
    ],
    "shirts": [
        "Men's slim fit cotton dress shirt white",
        "Women's silk blouse with tie neck",
        "Classic plaid flannel shirt red and black",
        "Plain crew neck t-shirt 100% cotton navy",
        "Short sleeve polo shirt pique cotton",
        "Oxford button-down shirt light blue",
        "Linen short sleeve summer shirt beige",
        "Graphic print band t-shirt black",
        "Long sleeve henley shirt heather grey",
        "Denim shirt western style with pearl snaps",
        "Striped breton top long sleeve",
        "Formal wing collar tuxedo shirt",
        "Kids' dinosaur print t-shirt",
        "V-neck fitted t-shirt pack of 3",
        "Checked lumberjack shirt brushed cotton",
        "Ladies' chiffon blouse floral print",
        "Rugby shirt striped heavyweight cotton",
        "Grandad collar shirt washed cotton",
        "Hawaiian print holiday shirt",
        "Thermal long sleeve base layer top",
        "Muscle fit stretch shirt black",
        "Oversized boyfriend fit shirt white",
        "Corduroy overshirt camel brown",
        "Sleeveless tank top ribbed cotton",
        "Baseball raglan t-shirt three-quarter sleeve",
        "Peter pan collar blouse cream",
        "Half-zip pullover shirt merino blend",
        "Cuban collar resort shirt palm print",
        "Pinstripe business shirt double cuff",
        "Tie-dye festival t-shirt multicolour",
        "Maternity wrap blouse jersey fabric",
        "Boys' school uniform shirts 2 pack",
        "Longline curved hem t-shirt olive",
        "Cropped blouse with puff sleeves",
        "Moisture-wicking sports training top",
        "Turtleneck long sleeve top fine knit",
        "Embroidered peasant blouse boho style",
        "Work utility shirt with chest pockets",
        "Slub cotton t-shirt garment dyed",
        "Satin pyjama-style shirt emerald green",
    ],
    "furniture": [
        "Oak dining table with 6 chairs",
        "3-seater fabric sofa charcoal grey",
        "King size bed frame with upholstered headboard",
        "Ergonomic office chair with lumbar support",
        "Solid pine wardrobe with 2 doors and drawer",
        "Glass top coffee table with metal legs",
        "5-tier bookshelf walnut finish",
        "Corner sofa with chaise left hand",
        "Memory foam mattress double 25cm deep",
        "Chest of drawers 5 drawer white",
        "Extendable dining table seats 8",
        "Leather recliner armchair brown",
        "TV stand with storage for 55 inch screens",
        "Standing desk height adjustable electric",
        "Bunk bed with ladder solid wood",
        "Velvet accent chair mustard yellow",
        "Bedside table with drawer and shelf",
        "Shoe storage cabinet slimline hallway",
        "Rattan garden furniture set 4 piece",
        "Console table narrow entryway oak",
        "Bar stools set of 2 adjustable height",
        "Futon sofa bed with wooden frame",
        "Kids' toy storage unit with canvas bins",
        "Dressing table with mirror and stool",
        "Filing cabinet 3 drawer lockable",
        "Nest of 3 side tables glass and chrome",
        "Ottoman storage bench faux leather",
        "Wall-mounted floating shelves set of 3",
        "Rocking chair nursery with padded cushion",
        "Sideboard buffet cabinet mid-century style",
        "Loft bed with desk underneath",
        "Chaise longue button tufted velvet",
        "Kitchen island trolley with butcher block top",
        "Single divan bed with storage drawers",
        "Gaming chair racing style with headrest",
        "Round pedestal dining table marble effect",
        "Hallway coat rack bench with shoe storage",
        "Recliner sofa 2 seater power operated",
        "Vanity desk with LED mirror white",
        "Outdoor patio table and chairs bistro set",
    ],
}


def build_dataframes(client):
    """Embed synthetic examples and category meanings, return the two collection dfs."""
    rows = [
        {"category": category, "example": example}
        for category, examples in CATEGORY_EXAMPLES.items()
        for example in examples
    ]
    df_samples = pd.DataFrame(rows)
    df_samples = add_gemini_embeddings(
        df=df_samples,
        text_column="example",
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_DOCUMENT",
        max_workers=8,
        batch_size=1,
        client=client,
    )

    df_avg = df_samples.groupby("category").agg({
        "embedding": lambda x: np.vstack(x).mean(axis=0).tolist(),
        "example": "first",
    }).reset_index()

    df_meanings = pd.DataFrame({
        "category": list(CATEGORY_MEANINGS.keys()),
        "meaning": list(CATEGORY_MEANINGS.values()),
    })
    df_meanings = add_gemini_embeddings(
        df=df_meanings,
        text_column="meaning",
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_DOCUMENT",
        max_workers=8,
        batch_size=1,
        client=client,
    )

    return df_meanings, df_avg


def write_collections(df_meanings, df_avg):
    """Replace both collections in the Chroma store."""
    chroma = PersistentClient(path=CHROMA_PATH)

    for name in ("intent_meaning_collection", "sample_avg_embeddings_collection"):
        try:
            chroma.delete_collection(name)
            print(f"Deleted existing collection: {name}")
        except Exception:
            print(f"No existing collection to delete: {name}")

    col_meaning = chroma.get_or_create_collection(
        name="intent_meaning_collection",
        metadata={"hnsw:space": "cosine"},
    )
    col_meaning.add(
        ids=[f"doc-{i}" for i in range(len(df_meanings))],
        embeddings=df_meanings["embedding"].tolist(),
        documents=df_meanings["meaning"].tolist(),
        metadatas=[{"output": c} for c in df_meanings["category"]],
    )
    print(f"Wrote intent_meaning_collection ({len(df_meanings)} items)")

    col_avg = chroma.get_or_create_collection(
        name="sample_avg_embeddings_collection",
        metadata={"hnsw:space": "cosine"},
    )
    col_avg.add(
        ids=[f"doc-{i}" for i in range(len(df_avg))],
        embeddings=df_avg["embedding"].tolist(),
        documents=df_avg["example"].tolist(),
        metadatas=[{"output": c} for c in df_avg["category"]],
    )
    print(f"Wrote sample_avg_embeddings_collection ({len(df_avg)} items)")


def main():
    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    df_meanings, df_avg = build_dataframes(client)
    write_collections(df_meanings, df_avg)

    # Sanity check: read back and print categories
    chroma = PersistentClient(path=CHROMA_PATH)
    for name in ("intent_meaning_collection", "sample_avg_embeddings_collection"):
        items = chroma.get_collection(name).get(include=["metadatas"])
        outputs = sorted({m["output"] for m in items["metadatas"]})
        print(f"{name}: {len(items['ids'])} items, categories: {outputs}")


if __name__ == "__main__":
    main()
