from supertokens_python import init, InputAppInfo, SupertokensConfig
from supertokens_python.recipe import thirdparty, session
import os

init(
    app_info=InputAppInfo(
        app_name="<YOUR_APP_NAME>",
        api_domain="<YOUR_API_DOMAIN>",
        website_domain="<YOUR_WEBSITE_DOMAIN>",
        api_base_path="/auth",
        website_base_path="/auth"
    ),
    supertokens_config=SupertokensConfig(
        # https://try.supertokens.com is for demo purposes. Replace this with the address of your core instance (sign up on supertokens.com), or self host a core.
        connection_uri=os.environ['SUPERTOKENS_CONNECTION_URI'],
        api_key=os.environ['SUPERTOKENS_API_KEY']
        # api_key=<API_KEY(if configured)>
    ),
    framework='flask',
    recipe_list=[
        session.init(), # initializes session features
        thirdparty.init(
           # TODO: See next step
        ) 
    ]
)