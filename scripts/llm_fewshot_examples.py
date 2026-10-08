few_shots = {
    # en_de_#CLIENT-01#_default_2020-12-20-12,10
    "ende_gemba": {
        "source_lang": "English",
        "source_seg": "I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.",
        "target_lang": "German",
        "target_seg": "Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.",
        "answer": """Critical:
no-error
Major:
accuracy/mistranslation - "involvement"
accuracy/omission - "the account holder"
Minor:
fluency/grammar - "wäre"
fluency/register - "dir"
""",
    },
    "ende_conversation": {
        "source_lang": "English",
        "source_seg": "As soon as we have heard back, tis is when you will be emailed",
        "target_lang": "German",
        "target_seg": "Sobald wir wieder gehört haben, tis ist, wenn Sie e-mail erhalten werden",
        "answer": """Critical:
accuracy/untranslated - "tis"
Major:
accuracy/mistranslation - "wieder gehört haben"
Minor:
accuracy/addition - "wenn"
source issue - "tis"
""",
    },
}


def mqm_fewshot(few_shots):
    prompts = [
        {
            "role": "system",
            "content": f"You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation.\n The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, source error or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension.",
        }
    ]

    template = """{source_lang} source:
```{source_seg}```
{target_lang} translation:
```{target_seg}```

Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them."""

    for shot in few_shots:
        prompts.append({"role": "user", "content": template.format(**shot)})
        answer = shot["answer"]

        prompts.append({"role": "assistant", "content": answer})

    prompts.append({"role": "user", "content": template})

    return prompts


# all context in source, source+hyp in the context
few_shots_context = {
    "ende_conversation": {
        "source_lang": "English",
        "source_seg": "As soon as we have heard back, tis is when you will be emailed",
        "target_lang": "German",
        "target_seg": "Sobald wir wieder gehört haben, tis ist, wenn Sie e-mail erhalten werden",
        "sender": "Agent",
        "context": """Customer (German): Hallo, ich habe einen Artikel, der letzte Woche geliefert werden sollte und der immer noch nicht geliefert wird.
Customer (German): Ich habe den Chat zweimal kontaktiert und bisher nichts.
Customer (German): Die First Lady sagte mir, dass es am Montag verschickt wird, die zweite Lady sagte mir gestern, dass sie eine E-Mail überprüfen und senden wird und sie hat nie geantwortet.
Customer (German): #ADDRESS#,
Agent (English): Good Morning #NAME#
Agent (English): Thanks for contacting #PRS_ORG#, you are through to #NAME#
Agent (English): So that I can assist you can you please provide your account details (Full Name, E-mail address, Postal Address and Order Number)
Customer (German): Bestellnummer: #NUMBER#
Customer (German): #ADDRESS#
Agent (English): Thank you - so this query is with warehouse as stated in yesterdays chat, we have to await the reply to the investigation.
""",
        "answer": """Critical:
accuracy/untranslated - "tis"
Major:
accuracy/mistranslation - "wieder gehört haben"
Minor:
accuracy/addition - "wenn"
source issue - "tis"
""",
    },
    "enfr_conversation": {
        "source_lang": "English",
        "source_seg": "Please sign out from the #PRS_ORG# account on the ereader and then wait 2 minutes to sign back in again.",
        "target_lang": "French",
        "target_seg": "S'il vous plaît vous désinscrire du compte #PRS_ORG# sur l'erreader et attendre 2 minutes pour vous inscrire à nouveau.",
        "sender": "Agent",
        "context": """Agent (English): Thank you for the information provided, I hope you are doing fine.
Agent (English): Please give me a moment.
Agent (English): Thank you for holding, I'm sorry you have issues doing a purchase.
Customer (French): Nous voulons acheter Etés anglais
Customer (French): et à chaque fois que nous confirmons le paiement
Agent (English): When was the last update of your billing information?
Customer (French): aujourd'hui car nous venons d'acheter la liseuse et de créer le compte
Agent (English): Ok.
""",
        "answer": """Critical:
no-error
Major:
accuracy/mistranslation - "désinscrire"
accuracy/mistranslation - "inscrire"
fluency/grammar - "attendre"
Minor:
fluency/spelling - "erreader"
style - "désinscrire"
""",
    },
    "enpt_conversation": {
        "source_lang": "English",
        "source_seg": "Unfortunately, we aren't able to book you, at this moment, but I've notified our Technical team of the issue so they can look into it ASAP.",
        "target_lang": "Portuguese",
        "target_seg": "infelizmente, não podemos reservar você, neste momento, mas eu tenho notificado a nossa equipe técnica do problema para que eles possam olhar para ele ASAP.",
        "sender": "Agent",
        "context": """Agent (English): I'm happy to check that for you!
Agent (English): Just one moment while I pull up your account :)
Agent (English): Sorry to keep you waiting, I'm having problem locating your account.
Customer (Portuguese): Sem problemas.
Customer (Portuguese):Precisa de alguma informação minha ?
Agent (English): Yes, how about an e-mail associate with us?
Customer (Portuguese): #EMAIL#
Agent (English): Let me try that, Thanks!
Agent (English): It looks like a technical error on the backend of #PRS_ORG# and Functional Training #PRS_ORG# schedule prevented you from booking this reservation.
""",
        "answer": """Critical:
accuracy/untranslated text - "ASAP"
Major:
accuracy/mistranslation - "reservar"
fluency/grammar - "tenho notificado"
fluency/grammar - "olhar para ele"
Minor:
style - "infelizmente"
""",
        },
    "enko_conversation": {
            "source_lang": "English",
            "source_seg": "Do you wish to escalate this?",
            "target_lang": "Korean",
            "target_seg": "이 사태를 더 악화시키고 싶니?",
            "sender": "Agent",
            "context": """Agent (English): I understand that your account is been implemented with some restrictions due to the violation of some rules.
Agent (English): Is that correct?
Customer (Korean): 네 하지만 저는 규칙을 위반한적이 없습니다
Customer (Korean): 제한을 풀어주십시요.
Customer (Korean): 저는 규칙을 위반하지않았습니다,
Agent (English): I'll try my best to help you with disputing the restrictions.
Agent (English): To begin, kindly share your registered email address or PRS-ORG ID of the account you concerning about for better assistance.
Customer (Korean): EMAIL
Customer (Korean): 제가 어떤규칙을 위반하였습니까?
Agent (English): Thank you for the code, I have checked your account and found that you have been banned In game for 7 days.
Agent (English): No worry, we have a dedicated team who works in these kinds of cases and can investigate properly.
Agent (English): I would like to inform you that this process might take 10 days to get complete.
""",
            "answer": """Critical:
accuracy/untranslated - "악화시키고"
Major:
fluency/register - "싶니?"
Minor:
terminology/inappropriate for context - "사태"
""",
        },
    "ennl_conversation": {
            "source_lang": "English",
            "source_seg": "nee dat niet ga zo eerst proberen de ereader op te laden en te down loaden.",
            "target_lang": "Dutch",
            "target_seg": "no, don't do that first, try to load the ereader first and then download.",
            "sender": "Agent",
            "context": """Agent (English): I am sorry that you are experiencing this issue, I will do my best to assist yo
Agent (English): if you have a second ereader you can sign in with your same PRS-ORG email address in the second ereader and all your library will be downloaded in the new ereader
Customer (Korean): het handigste voor mij is als u het in een email wil zetten EMAIL
Customer (Korean): oke dat ga ik later proberen
Agent (English): i have checked your account and these 2 email addresses are linked, you can use one or another, no problem
Agent (English): all your books are stored in the 2 email addresses
Agent (English): yes, you can try that later, For your information, I will be sending you a transcript of our conversation.
Agent (English): Should you have any further questions or concerns, you can always reply back to that email and we will be able to assist you further.
Agent (English): at the moment, is there any other doubt or request you may have?
Customer (Dutch): oke bedankt zover.
""",
            "answer": """Critical:
no-error
Major:
accuracy/mistranslation - "don't do that first"
accuracy/mistranslation - "load"
Minor:
fluency/inconsistency - "first"
""",
        },
    # en-zh: written for this repository (NOT taken from the human MQM annotations of the WMT24 chat task like the other pairs):
    # one major accuracy error ("再次登出" = sign out again, should be "重新登录") and one minor style error (redundant "之后，然后").
    "enzh_conversation": {
        "source_lang": "English",
        "source_seg": "Please sign out of your account on the e-reader, wait two minutes, and then sign back in.",
        "target_lang": "Chinese",
        "target_seg": "请在电子阅读器上退出您的账户，等待两分钟之后，然后再次登出。",
        "sender": "Agent",
        "context": """Customer (Chinese): 你好，我昨天买了一本电子书，但是在阅读器上找不到。
Agent (English): Hello, thank you for contacting us. I'm sorry to hear that.
Agent (English): Could you tell me which e-reader model you are using?
Customer (Chinese): 是最新的那款，买了大概两个月。
Agent (English): Thank you. Let me check the account details for you.
""",
        "answer": """Critical:
no-error
Major:
accuracy/mistranslation - "再次登出"
Minor:
style/awkward - "之后，然后"
""",
    },
}


def mqm_fewshot_context(few_shots):
    prompts = [
        {
            "role": "system",
            "content": f"You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation.\n The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, source error or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension.",
        }
    ]

    template = """Context:
```{context}```
{sender} source ({source_lang}):
```{source_seg}```
{target_lang} translation:
```{target_seg}```

Based on the conversation context between the agent and the customer, the current source by "{sender}" in {source_lang} and its machine translation in {target_lang} surrounded with triple backticks, identify error types in the translation and classify them."""

    for shot in few_shots:
        prompts.append({"role": "user", "content": template.format(**shot)})
        answer = shot["answer"]

        prompts.append({"role": "assistant", "content": answer})

    prompts.append({"role": "user", "content": template})

    return prompts


TEMPLATE_GEMBA_MQM_1shot = lambda x: mqm_fewshot([few_shots[x]])
TEMPLATE_GEMBA_CONTEXT_MQM_1shot = lambda x: mqm_fewshot_context([few_shots_context[x]])
