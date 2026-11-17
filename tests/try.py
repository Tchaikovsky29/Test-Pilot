from test_pilot import BrowserSession
from test_pilot.actions import Actor, Click
from test_pilot.policy import Policy

url = "https://phptravels.net/"
with BrowserSession() as s:
    s.goto(url)
    a = Actor(s, Policy(start_url=url))
    print(a.act(Click(id=89)).message)
    print(a.look())
    print([st.code() for st in a.steps])