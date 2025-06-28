#!/usr/bin/env python
# coding: utf-8

# In[3]:


from fastapi import FastAPI, HTTPException, Request, Response 
from typing import Optional
from pydantic import BaseModel

# Import your text processing function from wherever it's defined
from markdown_generator import process_text

app = FastAPI()

class TextPayload(BaseModel):
    main_text: str
    sub_text: Optional[str] = ""

@app.post("/process_texts/", response_class=Response)
def process_texts(payload: TextPayload):
    try:
        result = process_text(payload.main_text, payload.sub_text)
        return Response(content=result, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

