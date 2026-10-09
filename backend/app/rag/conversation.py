from __future__ import annotations

import threading
from collections import OrderedDict


class ConversationStore:
    def __init__(self, max_turns: int = 6, max_conversations: int = 500):
        self.max_turns = max_turns
        self.max_conversations = max_conversations
        self._data: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self._lock = threading.Lock()

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._data.get(conversation_id, []))

    def add_turn(self, conversation_id: str, question: str, answer: str) -> None:
        with self._lock:
            history = self._data.setdefault(conversation_id, [])
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            del history[: max(0, len(history) - self.max_turns * 2)]
            self._data.move_to_end(conversation_id)
            while len(self._data) > self.max_conversations:  # evict least recently used
                self._data.popitem(last=False)

    def clear(self, conversation_id: str) -> None:
        with self._lock:
            self._data.pop(conversation_id, None)
