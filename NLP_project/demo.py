import time
from train import Config
from utils import load_model, bleu_score, Translate


def test():
    config = Config()
    models = [load_model(config.model_path[i], config) for i in range(len(config.model_path))]

    texts = [
        {
            "input": "鲁监门之女婴相从绩，中夜而泣涕，其偶曰：“何谓而泣也？",
            "output": "鲁国监门人的女儿婴随人一起绩麻，半夜哭泣起来，她同伴问她：“为什么哭泣？”",
        },
        {
            "input": "温恒云“京口酒可饮，兵可用”，深不欲愔居之。",
            "output": "桓温常常说“京口的酒可以喝，士兵可以任用”，内心非常不希望郗愔留在那里。"
        },
        {
            "input": "到兰于山南以分单于兵，毋令专乡贰师军。",
            "output": "我希望能够独立带领一队，到兰干山南分散单于的兵力，不要让匈奴专门针对贰师将军的部队。"
        },
        {
            "input": "却军还众，不犯魏境者，贤干木之操，高魏文之礼也。",
            "output": "秦国退兵还师，不进犯魏国边境的原因，是看重段干木的操守，推崇魏文侯的重礼。"
        },
        {
            "input": "吾求公数岁，公辟逃我，今公何自从吾儿游乎？",
            "output": "我找你们好几年，你们躲着不见我，现在你们为什么来跟我儿子交往呢？"
        },
        {
            "input": "为将而降，降而为之效死以战，虽欲浣涤其污，而已缁之素，不可复白。",
            "output": "身为将领却投降敌军，投降后又为敌方拼死作战，即使想洗刷污名，也如同染黑的白布无法复原。"
        },
        {
            "input": "吾群臣无有不骄侮之意者，唯赫子不失君臣之礼，是以先之。",
            "output": "我的大臣们都对我有高傲轻慢的意思，只有高赫没有失掉君臣之间的礼节，所以先奖赏他。"
        },
    ]

    print('-' * 100)
    for text in texts:
        now_time = time.time()
        translated_text = Translate(text, models, config)[0]
        print("原文：", text["input"])
        print("译文：", text["output"])
        print("模型翻译：", translated_text)
        print(bleu_score(translated_text, text["output"]), f"\ttime: {time.time() - now_time :.2f}s",)
        print("-" * 100)



def demo(text):
    config = Config()
    models = [load_model(config.model_path[i], config) for i in range(len(config.model_path))]
    translated_text = Translate(text, models, config)
    print(f"原文：{text}")
    print(f"译文：{translated_text}")



if __name__ == '__main__':
    test()
    # demo("为将而降，降而为之效死以战，虽欲浣涤其污，而已缁之素，不可复白。")
