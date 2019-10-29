from typing import List
from json import JSONDecodeError
from aiohttp import web
from deeppavlov import build_model, configs


#rusentiment_bert work powerfull with russian language
model = build_model(configs.classifiers.rusentiment_bert, download=False)

app = web.Application(debug=False)

sent = dict(
    negative=-1,
    positive=1,
    neutral=0,
    skip=2,
    speech=3,
)


async def sentiment_analysis(request: web.Request):
    try:
        body: List[dict] = await request.json()
    except JSONDecodeError:
        raise web.HTTPNoContent(reason= "No content to decode!")

    result = []

    while body:
        data = {}
        itm_body: dict = body.pop()

        text = itm_body.get("text")
        text_id = itm_body.get("id")

        try:
            assert text is not None and text_id is not None
        except AssertionError:
            raise web.HTTPBadRequest(reason="Value mismatch!")

        sentiment = model([text])

        try:
            assert sentiment[0] in sent
        except AssertionError:
            raise web.HTTPExpectationFailed(reason=f'Unknown sentiment {sentiment[0]}!')

        data["id"] = text_id
        data["text"] = sent.get(sentiment[0])
        result.append(data)
    return web.json_response(sorted(result, key=lambda i:i["id"]))


app.add_routes([
    web.post('/run', sentiment_analysis),
])


if __name__=='__main__':
    web.run_app(app)