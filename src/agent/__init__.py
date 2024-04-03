from .rad import RAD
from .curl import CURL
from .sac_ae import SACAE
from .sac import SAC
from .drq import DRQ
from .atc import ATC
from .curl_conservative import CURL_Conservative

Agent_Config = {
    'curl': CURL,
    'sacae': SACAE,
    'sac': SAC,
    'rad': RAD,
    'drq': DRQ,
    'atc': ATC,
    'curl_conservative' : CURL_Conservative
}

def make_agent(model, device, action_shape, args):
    Agent = Agent_Config[args.agent]
    return Agent(model, device, action_shape, args)
