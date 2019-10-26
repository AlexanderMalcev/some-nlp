from aiohttp import web
from deeppavlov import build_model, configs


model = build_model(configs.classifiers.rusentiment_bert)

app = web.Application(debug=True)
sent = dict(
    negative=-1,
    positive=1,
    neutral=0,
    skip=0,
    speech=2,
)

async def sentiment_analysis(request: web.Request):
    body: dict = await request.json()
    text_id = body.get('id')
    text = body.get('text')
    try:
        assert text_id is not None and text is not None
    except AssertionError:
        raise web.HTTPBadRequest(reason='Value mismatch')
    result = model([text])
    print(result)
    try:
        assert result[0] in sent 
    except AssertionError:
        raise web.HTTPExpectationFailed(reason=f'Unknown sentiment {result[0]}!')
    return web.json_response(
        {
            'id':text_id,
            'sentiment':sent.get(result[0])
        }
    )


app.add_routes([
    web.post('/run', sentiment_analysis),
])


if __name__=='__main__':
    web.run_app(app)