import asyncio
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError

load_dotenv()

API_ID = int(os.getenv("TELEGRAM_API_ID"))
API_HASH = os.getenv("TELEGRAM_API_HASH")
CHANNEL = os.getenv("TELEGRAM_CHANNEL", "nerdsmedia")


async def fetch_with_retry(client, channel, limit=10000, max_retries=5):
    for attempt in range(max_retries):
        try:
            return await client.get_messages(channel, limit=limit)
        except FloodWaitError as e:
            wait = e.seconds + 5
            print(f"FloodWait: спим {wait} с (попытка {attempt + 1}/{max_retries})")
            await asyncio.sleep(wait)
    raise RuntimeError("FloodWait: лимит попыток исчерпан")


async def parse_channel():
    async with TelegramClient("session", API_ID, API_HASH) as client:
        return await fetch_with_retry(client, CHANNEL)


def messages_to_dataframe(messages):
    rows = []
    for msg in messages:
        if not msg.text:
            continue
        reactions = sum(r.count for r in msg.reactions.results) if msg.reactions else 0
        comments = msg.replies.replies if msg.replies else 0
        rows.append(
            {
                "id_post": msg.id,
                "date": msg.date,
                "text": msg.text,
                "views": msg.views,
                "reactions": reactions,
                "comments": comments,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    messages = asyncio.run(parse_channel())
    df = messages_to_dataframe(messages)
    out = Path("data") / "nerds.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} posts to {out}")
