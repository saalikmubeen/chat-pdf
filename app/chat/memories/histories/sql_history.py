from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from app.web.api import (
    get_messages_by_conversation_id,
    add_message_to_conversation
)


class SqlMessageHistory(BaseChatMessageHistory):
    """Custom chat message history backed by SQL database."""

    def __init__(self, conversation_id: str):
        """Initialize with conversation ID."""
        super().__init__()
        self.conversation_id = conversation_id

    @property
    def messages(self) -> list[BaseMessage]:
        """Retrieve messages from database."""
        messages_list = get_messages_by_conversation_id(self.conversation_id)
        # Ensure we always return a list that can be copied
        if not messages_list:
            return []
        # Return a fresh list copy to avoid any reference issues
        return [msg for msg in messages_list]

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the database."""
        add_message_to_conversation(
            conversation_id=self.conversation_id,
            role=message.type,
            content=message.content
        )

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add multiple messages to the database."""
        for message in messages:
            self.add_message(message)

    def clear(self) -> None:
        """Clear conversation history (implement if needed)."""
        pass
