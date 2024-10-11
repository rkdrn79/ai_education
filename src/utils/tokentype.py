from enum import Enum

class TokenType(int, Enum):
    TEXT = 0
    IMAGE = 1

    @staticmethod
    def to_str(token_type: "TokenType") -> str:
        return {
            TokenType.TEXT: 'text',
            TokenType.IMAGE: 'image',
        }[token_type]
    
    @staticmethod
    def to_int(token_type: str) -> "TokenType":
        return {
            'text': TokenType.TEXT,
            'image': TokenType.IMAGE,
        }[token_type]