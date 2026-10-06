"""
Run this once on your own computer to get STRING_SESSION for your own Telegram account.
    pip install pyrotgfork tgcrypto
    python gen_session.py
Never share the printed string with anyone.
"""
import asyncio

from pyrogram import Client


async def main():
    api_id = int(input("API_ID: ").strip())
    api_hash = input("API_HASH: ").strip()
    async with Client("gen", api_id=api_id, api_hash=api_hash, in_memory=True) as app:
        print("\nSTRING_SESSION:\n")
        print(await app.export_session_string())


asyncio.run(main())
