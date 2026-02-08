# from pydantic import BaseModel, Extra


# class Metadata(BaseModel, extra=Extra.allow):
#     conversation_id: str
#     user_id: str
#     pdf_id: str


# class ChatArgs(BaseModel, extra=Extra.allow):
#     conversation_id: str
#     pdf_id: str
#     metadata: Metadata
#     streaming: bool


from pydantic import BaseModel, ConfigDict


class Metadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    conversation_id: str
    user_id: str
    pdf_id: str

    # class Config:
    #     extra = "allow"  # Updated for Pydantic v2


class ChatArgs(BaseModel):
    model_config = ConfigDict(extra="allow")

    conversation_id: str
    pdf_id: str
    metadata: Metadata
    streaming: bool = False

    # class Config:
    #     extra = "allow"  # Updated for Pydantic v2
