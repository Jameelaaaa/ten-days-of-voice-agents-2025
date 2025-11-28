import logging
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # backend/src
BACKEND_ROOT = os.path.dirname(BASE_DIR)               # backend
ORDER_DIR = os.path.join(BACKEND_ROOT, "order")

# Load catalog (recipes included in catalog.json)
with open(os.path.join(BACKEND_ROOT, "shared-data", "catalog.json"), encoding="utf-8") as f:
    catalog = json.load(f)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant for Aura Basket. "
                "You help users order groceries and simple meal ingredients.\n\n"
                "CART BEHAVIOR:\n"
                "- When the user says things like 'add orange juice', 'I want KitKat', or "
                "'add 2 pizzas to my cart', call add_item_tool.\n"
                "- Pass a short, lowercase phrase describing the item (for example "
                "'orange juice', 'kitkat', 'veg pizza'). The tool will map this to the "
                "correct internal id from the catalog.\n"
                "- When the user says 'remove X from my cart', call remove_item_tool.\n"
                "- When the user asks 'what is in my cart' or 'show my cart', call show_cart_tool.\n"
                "- When the user asks for ingredients like 'pizza ingredients', "
                "'ingredients for sandwich', or 'milkshake ingredients', call add_recipe_tool "
                "with the recipe name that best matches the dish (for example 'Veg Pizza', "
                "'Veg Sandwich', 'Milkshake').\n"
                "- When the user asks 'what is the total', first call show_cart_tool and then clearly "
                "say the total in Indian Rupees.\n\n"
                "ORDER PLACEMENT FLOW (IMPORTANT):\n"
                "1) When the user says 'place my order', 'that's all', or 'I'm done', DO NOT place the order immediately.\n"
                "2) First ask: 'Before I place your order, may I know the name for the order?'\n"
                "3) When the user answers with a name, repeat it back and ask for confirmation, for example: "
                "'So your name is Rahul, should I place the order now?'\n"
                "4) Only after the user confirms, call place_order_tool with that exact name.\n"
                "5) After placing the order, tell the user that the order is placed and the total amount in Rupees.\n\n"
                "FAREWELL BEHAVIOR:\n"
                "- When the user says 'thank you', 'thanks', 'bye', 'goodbye' or similar, call farewell_tool.\n"
                "- The farewell reply must be given as multiple separate sentences (one short sentence per idea), "
                "for example: 'Enjoy. Do visit Aura Grocery Store again. Thank you — have a great day.'\n\n"
                "All items are vegetarian. Prices are in Indian Rupees (₹). "
                "Always explain clearly what you did with the cart and what the current total is when asked."
            ),
        )
        self.cart: list[dict] = []
        self.pending_name: str | None = None

    #Search the catalog 
    def _iter_items(self):
        """List all items from all categories."""
        for category in catalog.get("categories", []):
            for item in category.get("items", []):
                yield item

    def _all_items_list(self) -> list[dict]:
        """Return list of all item dicts - convenience helper."""
        return list(self._iter_items())

    def _find_item_id_from_text(self, text: str) -> str | None:
        """
        Map natural text like 'orange juice', 'kit kat', 'pizza', or 'amul veg pizza'
        to the closest item id.
        """
        if not text:
            return None
        t_orig = text.strip()
        t = t_orig.lower().strip()
        t_compact = re.sub(r"[\s_-]+", "", t)  # remove spaces/underscores/dashes

        # collect candidates
        items = self._all_items_list()

        # 1) exact id match
        for item in items:
            if str(item.get("id", "")).lower() == t:
                return item["id"]

        # 2) exact name match
        for item in items:
            if str(item.get("name", "")).lower() == t:
                return item["id"]

        # 3) compact name/id match
        for item in items:
            item_id = str(item.get("id", "")).lower()
            name_compact = re.sub(r"[\s_-]+", "", str(item.get("name", "")).lower())
            if t_compact == item_id or t_compact == name_compact:
                return item["id"]

        # 4) contains / substring match (prefer name matches)
        for item in items:
            name = str(item.get("name", "")).lower()
            item_id = str(item.get("id", "")).lower()
            if t in name or t in item_id:
                return item["id"]

        # 5) token match: all words present in name
        tokens = re.findall(r"\w+", t)
        if tokens:
            for item in items:
                name = str(item.get("name", "")).lower()
                if all(tok in name for tok in tokens):
                    return item["id"]

        # no match
        return None

    def _list_category(self, category_name: str) -> str:
        """
        Return a readable list of items in a category, or a helpful message if category not found.
        """
        if not category_name:
            return "Please tell me which category you want to see (for example: 'snacks' or 'prepared food')."

        c = None
        requested = category_name.strip().lower()
        # find best matching category by name (case-insensitive, substring)
        for cat in catalog.get("categories", []):
            cat_name = str(cat.get("name", "")).lower()
            if requested == cat_name or requested in cat_name or cat_name in requested:
                c = cat
                break

        if not c:
            # try plural/singular normalization (simple)
            for cat in catalog.get("categories", []):
                cat_name = str(cat.get("name", "")).lower()
                if requested.rstrip("s") == cat_name.rstrip("s"):
                    c = cat
                    break

        if not c:
            return f"I don't have a category named '{category_name}'. Available categories: " + ", ".join(
                [str(cat.get("name", "")) for cat in catalog.get("categories", [])]
            )

        items = c.get("items", [])
        if not items:
            return f"There are no items listed in the {c.get('name')} category."

        lines = [f"- {it.get('name')} (id: {it.get('id')}, price: ₹{it.get('price', '?')}, unit: {it.get('unit','?')})" for it in items]
        return f"Items in {c.get('name')}:\n" + "\n".join(lines)

    # Internal cart logic

    def _add_item(self, item_id_or_name: str, quantity: int = 1) -> str:
        resolved_id = self._find_item_id_from_text(item_id_or_name)
        if not resolved_id:
            return f"Item '{item_id_or_name}' not found in Aura Basket catalog."

        for item in self._iter_items():
            if item.get("id") == resolved_id:
                try:
                    q = int(quantity)
                    if q < 1:
                        q = 1
                except Exception:
                    q = 1
                self.cart.append(
                    {
                        "id": item["id"],
                        "name": item["name"],
                        "quantity": q,
                        "price": item.get("price", 0),
                        "unit": item.get("unit", ""),
                    }
                )
                return f"Added {q} x {item['name']} to your cart."
        return f"Item '{item_id_or_name}' not found in Aura Basket catalog."

    def _remove_item(self, item_id_or_name: str) -> str:
        resolved_id = self._find_item_id_from_text(item_id_or_name) or item_id_or_name
        before = len(self.cart)
        self.cart = [item for item in self.cart if item["id"] != resolved_id]
        if len(self.cart) < before:
            return "Item removed from your cart."
        return "That item was not in your cart."

    def _show_cart(self) -> str:
        if not self.cart:
            return "Your cart is empty."
        lines = [
            f"- {item['quantity']} x {item['name']} ({item['unit']})"
            for item in self.cart
        ]
        total = sum(item["quantity"] * item.get("price", 0) for item in self.cart)
        return "Items in your cart:\n" + "\n".join(lines) + f"\nTotal: ₹{total}"

    def _add_recipe(self, recipe_name: str) -> str:
        if not recipe_name:
            return "Please tell me which recipe you want (for example: 'pizza' or 'veg sandwich')."

        recipe_name_norm = recipe_name.strip().lower()
        recipes_cat = None
        for cat in catalog.get("categories", []):
            if cat.get("name", "").lower() == "recipes":
                recipes_cat = cat
                break

        if not recipes_cat:
            return "No recipes available in the catalog."

        chosen_recipe = None
        for recipe in recipes_cat.get("items", []):
            if recipe.get("name", "").lower() == recipe_name_norm:
                chosen_recipe = recipe
                break
            if str(recipe.get("id", "")).lower() == recipe_name_norm:
                chosen_recipe = recipe
                break

        if not chosen_recipe:
            for recipe in recipes_cat.get("items", []):
                if recipe_name_norm in recipe.get("name", "").lower():
                    chosen_recipe = recipe
                    break

        if not chosen_recipe:
            for cat in catalog.get("categories", []):
                for it in cat.get("items", []):
                    if recipe_name_norm in str(it.get("name", "")).lower() or recipe_name_norm == str(it.get("id", "")).lower():
                        candidate_id = str(it.get("id", "")).lower()
                        for recipe in recipes_cat.get("items", []):
                            if str(recipe.get("id", "")).lower() == candidate_id:
                                chosen_recipe = recipe
                                break
                    if chosen_recipe:
                        break
                if chosen_recipe:
                    break

        if not chosen_recipe:
            return f"Recipe for '{recipe_name}' not found. Available recipes: " + ", ".join(
                [r.get("name", "") for r in recipes_cat.get("items", [])]
            )

        ingredient_ids = chosen_recipe.get("ingredients", [])
        if not ingredient_ids:
            return f"The recipe '{chosen_recipe.get('name')}' does not list any ingredients."

        added = []
        not_found = []
        for ingr in ingredient_ids:
            found_id = None
            if self._find_item_id_from_text(ingr):
                found_id = self._find_item_id_from_text(ingr)
            else:
                alt = ingr.replace("_", " ").replace("-", " ")
                found_id = self._find_item_id_from_text(alt)

            if found_id:
                _ = self._add_item(found_id, quantity=1)
                for it in self._iter_items():
                    if it.get("id") == found_id:
                        added.append(it.get("name"))
                        break
            else:
                not_found.append(ingr)

        response_parts = []
        if added:
            response_parts.append(f"I added these ingredients to your cart: {', '.join(added)}.")
        if not_found:
            nf_readable = ", ".join(not_found)
            response_parts.append(
                f"However, I couldn't find these ingredients in the catalog: {nf_readable}. "
                "You can add similar items by name if you want."
            )

        return " ".join(response_parts)

    def _place_order(self, customer_name: str) -> str:
        order = {
            "items": self.cart,
            "total": sum(item["quantity"] * item.get("price", 0) for item in self.cart),
            "timestamp": datetime.now().isoformat(),
            "customer": customer_name,
        }

        os.makedirs(ORDER_DIR, exist_ok=True)

        def _sanitize_filename(s: str) -> str:
            s = s.strip().lower()
            return re.sub(r"[^a-z0-9_-]", "_", s)

        safe_name = _sanitize_filename(customer_name) or "customer"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{safe_name}_{ts}.json"
        order_path = os.path.join(ORDER_DIR, base_filename)

        suffix = 1
        final_path = order_path
        while os.path.exists(final_path):
            final_path = os.path.join(ORDER_DIR, f"{safe_name}_{ts}_{suffix}.json")
            suffix += 1

        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(order, f, indent=4, ensure_ascii=False)

        total = order["total"]
        self.cart = []
        return f"Your order has been placed and saved as {os.path.basename(final_path)}. The total is ₹{total}."

    #Farewell behavior 

    def _farewell_reply(self) -> str:
        """
        Return a multi-sentence farewell. Each idea is its own short sentence
        so Gemini-style output keeps them separated.
        """
        sentences = [
            "Enjoy.",
            "Do visit Aura Basket again.",
            "Thank you for shopping with us.",
            "Have a great day."
        ]
        # join with a space so result is a normal paragraph but still contains separate sentences
        return " ".join(sentences)

    # Tools 

    @function_tool
    async def add_item_tool(self, ctx: RunContext, item_text: str, quantity: int = 1) -> str:
        return self._add_item(item_text, quantity)

    @function_tool
    async def remove_item_tool(self, ctx: RunContext, item_text: str) -> str:
        return self._remove_item(item_text)

    @function_tool
    async def show_cart_tool(self, ctx: RunContext) -> str:
        return self._show_cart()

    @function_tool
    async def add_recipe_tool(self, ctx: RunContext, recipe_name: str) -> str:
        return self._add_recipe(recipe_name)

    @function_tool
    async def place_order_tool(self, ctx: RunContext, customer_name: str) -> str:
        return self._place_order(customer_name)

    @function_tool
    async def list_category_tool(self, ctx: RunContext, category_name: str) -> str:
        return self._list_category(category_name)

    @function_tool
    async def farewell_tool(self, ctx: RunContext) -> str:
        """
        Return the multi-sentence farewell reply. Call this when the user says 'thank you' or 'bye'.
        """
        return self._farewell_reply()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))




