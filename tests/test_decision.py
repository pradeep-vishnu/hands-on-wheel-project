from howvision.inference.adapters import RuleClassifier
from howvision.inference.types import ContactFeatures,WheelDetection
def test_unknown_reachable_without_wheel():assert RuleClassifier().predict(ContactFeatures(2,2,0,0,0),[],WheelDetection(False,0,[])).state=='UNKNOWN'
