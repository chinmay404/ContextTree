import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.agent.store.PostgresStore import PostgresConversationStore

MESSAGE_IDS = [
    "prx4f3nsmks7jk1z",
    "u5of485lmks7h22e",
]


def main() -> None:
    store = PostgresConversationStore()

    for message_id in MESSAGE_IDS:
        print("=" * 80)
        print(f"Message ID: {message_id}")
        message = store.get_message_by_id(message_id)
        if not message:
            print("Not found in DB.")
            recent = store.get_recent_messages(5)
            if recent:
                print("Recent messages in DB:")
                for idx, msg in enumerate(recent, start=1):
                    print(
                        f"{idx}. [{msg.get('role')}] {msg.get('message_id')} | thread={msg.get('thread_id')} | {msg.get('text')}"
                    )
            continue

        thread_id = message.get("thread_id")
        print(f"Thread ID: {thread_id}")
        print(f"Role: {message.get('role')}")
        print(f"Text: {message.get('text')}")
        print(f"Timestamp: {message.get('timestamp')}")

        summary = store.get_thread_summary(None, thread_id)
        print("Summary:")
        print(summary if summary else "<empty>")

        _, history = store.get_messages_until(None, thread_id, message_id)
        print("\nHistory up to this message:")
        for idx, msg in enumerate(history, start=1):
            print(f"{idx}. [{msg.get('role')}] {msg.get('text')}")


if __name__ == "__main__":
    main()
