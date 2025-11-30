import {ChatGoogleGenerativeAI} from "@langchain/google-genai";
import {
  Annotation,
  END,
  messagesStateReducer, START,
  StateGraph
} from "@langchain/langgraph";
import {AIMessage, HumanMessage, SystemMessage} from "@langchain/core/messages";

const model = new ChatGoogleGenerativeAI({model: "gemini-2.5-flash-lite"})
const annotation = Annotation.Root({
  messages: Annotation({reducer: messagesStateReducer, default: () => []}),
})
const generatePrompt = new SystemMessage(
    `넌 훌륭한 한 단락의 에세이를 작성하는 작가야.
    사용자 요청에 맞추어 최상의 에세이를 작성해.
    사용자가 비평을 제공하면, 수정해서 응답해.`
)

async function generate(state) {
  const answer = await model.invoke([generatePrompt, ...state.messages])
  return { messages: [answer] }
}

const reflectionPrompt = new SystemMessage(
    `넌 에세이 비평가야. 독자 입장에서 흐름상 어색하거나, 이해하기 어려운 내용이 있는지 알려줘`
)

async function reflect(state) {
  // AI 메시지를 인간 메시지로, 인간 메시지를 AI 메시지로 치환함
  // AI는 인간 - AI 메시지 쌍으로 파인튜닝 되어 있다는 점 이용
  const clsMap = {
    ai: HumanMessage,
    human: AIMessage,
  }
  const translated = [
      reflectionPrompt, state.messages[0], ...state.messages.slice(1).map(m => new clsMap[m.type](m.content))
  ]
  const answer = await model.invoke(translated)
  return { messages: [new HumanMessage(answer.content)] }
}

function shouldContinue(state) {
  if (state.messages.length > 6) { // 3회 실행
    return END;
  } else {
    return 'reflect'
  }
}

const builder = new StateGraph(annotation)
.addNode('generate', generate)
.addNode('reflect', reflect)
.addEdge(START, 'generate')
.addConditionalEdges('generate', shouldContinue)
.addEdge('reflect', 'generate')
const graph = builder.compile()

const input = {
  messages: [
    new HumanMessage(`
      숙소는 언덕에 있었다.
      어려서부터 산에 살았던지라 오르막에는 익숙하긴 했지만 피곤했던 탓인지 캐리어를 같이 끌어 올려야 해서인지 실제보다 더 언덕같은 느낌이었다.
      숙소는 별도의 커뮤니티 공간에 있었다.
      커다란 대문이 하나 있고 그 안에 다른 빌라들과 함께 모여 있는 형태였다.
      대문 여는 방법을 몰라 어떻게 열어야 하나 찾아보다가 누군가 음식 배달을 시켰는지 배달 기사가 도착했다.
      하지만 그 역시 문을 여는 방법을 모르는건 마찬가지였다.
    `.trim().split('\n').map(s => s.trim()).join(' '))
  ],
}
for await (const chunk of await graph.stream(input)) {
  console.log(chunk)
}

// 귀찮았는지 1회 수정 이후에 수정하지 않는다. 매끄러운 수정을 위해 프롬프트 수정 + 메시지 필터링이 필요할 것 같다.
/*
{
  generate: {
    messages: [
      AIMessage {
        "id": "f8394471-50db-4e6a-89c3-d0d525f0efa8",
        "content": "이 숙소는 약간의 도전 의식을 불러일으키는 곳이었다. 언덕에 자리 잡고 있어 어린 시절 산에서 보낸 시간을 떠올리게 했지만, 피곤함과 묵직한 캐리어 때문인지 실제보다 훨씬 더 가파르게 느껴졌다. 숙소는 다른 빌라들 과 함께 하나의 커뮤니티 단지에 위치해 있었는데, 입구를 알리는 큰 대문은 굳게 닫혀 있었다. 어떻게 열어야 할지 몰라 잠시 망설이던 찰나, 때마침 음식 배달 기사가 도착했지만 그 역시 대문을 여는 방법을 알지 못해 나와 같은 어려움을 겪고 있었다.",
        "additional_kwargs": {
          "finishReason": "STOP",
          "index": 0,
          "__gemini_function_call_thought_signatures__": {}
        },
        "response_metadata": {
          "tokenUsage": {
            "promptTokens": 184,
            "completionTokens": 143,
            "totalTokens": 327
          },
          "finishReason": "STOP",
          "index": 0
        },
        "tool_calls": [],
        "invalid_tool_calls": [],
        "usage_metadata": {
          "input_tokens": 184,
          "output_tokens": 143,
          "total_tokens": 327
        }
      }
    ]
  }
}
{
  reflect: {
    messages: [
      HumanMessage {
        "id": "4654bd81-668b-4493-a048-e0bc1115d711",
        "content": "네, 에세이 비평가로서 독자의 입장에서 흐름이 어색하거나 이해하기 어려운 부분을 살펴보겠습니다.\n\n**전반적인 인상:**\n\n제시해주신 두 가지 버전 모두 숙소의 위치와 도착 과정을 설명하고 있습니다. 첫 번째 버전 은 조금 더 구어체적이고 일상적인 느낌을 주는 반면, 두 번째 버전은 좀 더 문학적이고 묘사적인 표현을 사용하고 있습니다. 어떤 톤을 의도하셨는지에 따라 평가가 달라질 수 있습니다.\n\n**첫 번째 버전 (구어체 느낌):**\n\n*   **\"어려서부터 산에 살았던지라 오르막에는 익숙하긴 했지만 피곤했던 탓인지 캐리어를 같이 끌어 올려야 해서인지 실제보다 더 언덕같은 느낌이었다.\"**\n    *   이 문장은 복합적인 이유를 나열하고 있어 아주 자연스럽지는 않습니다. '익숙하긴 했지만'이라는 표현과 뒤따르는 이유들이 약간 꼬리를 무는 느낌을 줍니다. 조금 더 간결하게 표현하면 좋을 것 같습니다. 예를 들어, \"어릴 적 산에 살아 오르막길은 익숙했지만, 피곤함과 캐리어 때문인지 유난히 더 힘들게 느껴졌다.\" 와 같이 표현할 수 있습니다.\n*   **\"숙소는 별도의 커뮤니티 공간에 있었다. 커다란 대문이 하나 있고 그 안에 다른 빌라들과 함께 모여 있는 형태였다.\"**\n    *   '별도의 커뮤니티 공간'이라는 표현이 약간 모호하게 느껴질 수 있습니다. '독립 적인 단지'나 '펜션 단지'와 같이 좀 더 구체적인 표현이 와닿을 수 있습니다.\n    *   두 번째 문장은 앞 문장에서 말한 '커뮤니티 공간'이 어떤 곳인지 설명해주지만, '모여 있는 형태였다'는 표현이 다소 밋밋하게 느껴집니다.\n*   **\"대 문 여는 방법을 몰라 어떻게 열어야 하나 찾아보다가 누군가 음식 배달을 시켰는지 배달 기사가 도착했다. 하지만 그 역시 문을 여는 방법을 모르는건 마찬가지였다.\"**\n    *   '어떻게 열어야 하나 찾아보다가'라는 부분이 약간 늘어지는 느낌이 있습니다.\n    *   '모르는건 마찬가지였다'는 표현은 틀린 것은 아니지만, 문맥상 조금 더 자연스럽게 연결할 수 있습니다.\n\n**두 번째 버전 (문학적/묘사적 느낌):**\n\n*   **\"이 숙소는 약간의 도전 의식을 불러일으키는 곳이었다.\"**\n    *   이 문장은 앞선 내용과 직접적으로 연결되지 않고 갑자기 등장합니다. 앞선 숙소의 위치 묘사(언덕, 단지)가 '도전 의식'을 불러일으킨다는 맥락을 조금 더 명확하게 설명해주거나, 아니면 이 문장은 나중에 배치하는 것이 좋을  것 같습니다.\n*   **\"언덕에 자리 잡고 있어 어린 시절 산에서 보낸 시간을 떠올리게 했지만, 피곤함과 묵직한 캐리어 때문인지 실제보다 훨씬 더 가파르게 느껴졌다.\"**\n    *   이 문장은 첫 번째 버전의 첫 문장을 훨씬 더 섬세하고 감성적으로 표현했습니다. '떠올리게 했지만'이라는 표현이 앞의 '도전 의식'과 연결될 수 있겠네요.\n*   **\"숙소는 다른 빌라들과 함께 하나의 커뮤니티 단지에 위치해 있었는데, 입구를 알리는 큰 대문은 굳게 닫혀 있었다.\"**\n    *   '입구 를 알리는 큰 대문'이라는 표현은 좋습니다. '굳게 닫혀 있었다'는 묘사도 상황을 잘 전달합니다.\n*   **\"어떻게 열어야 할지 몰라 잠시 망설이던 찰나, 때마침 음식 배달 기사가 도착했지만 그 역시 대문을 여는 방법을 알지 못해 나와 같은 어려움을 겪고 있었다.\"**\n    *   '잠시 망설이던 찰나'는 좋은 표현입니다.\n    *   '그 역시 대문을 여는 방법을 알지 못해 나와 같은 어려움을 겪고 있었다'는 부분은 조금 더 부드럽게 연결될 수 있습니다. 예를 들어, \"그 역시 나와  마찬가지로 대문을 열 방법을 몰라 잠시 머뭇거리고 있었다.\" 와 같이 표현할 수 있습니다.\n\n**종합적인 제안:**\n\n두 가지 버전을 섞어서 더 나은 글을 만들 수도 있습니다.\n\n*   **시작:**\n    *   \"숙소는 언덕에 자리 잡고 있었다. 어린 시절 산에서 보낸 시간을 떠올리게 하는 익숙한 풍경이었지만, 피곤함과 묵직한 캐리어 때문인지 평소보다 훨씬 가파르게 느껴졌다.\" (두 번째 버전의 묘사를 활용)\n*   **단지 설명:**\n    *   \"숙소는 다른 빌라들과 함께 단지를 이 루고 있었고, 외부로 나가는 입구에는 커다란 대문이 굳게 닫혀 있었다.\" (두 버전의 핵심 내용을 통합)\n*   **도착 과정:**\n    *   \"대문을 어떻게 열어야 할지 몰라 잠시 망설이고 있을 때, 때마침 음식 배달 기사가 도착했다. 하지만 그 역시 대문을 열 방법을 몰라 나와 같은 난처함을 겪고 있었다.\" (자연스러운 연결)\n\n**결론적으로,**\n\n*   **흐름상 어색한 부분:** 첫 번째 버전은 문장 간 연결이 조금 더 매끄러울 수 있고, 두 번째 버전은 첫 문장의 맥락이 조금 더  명확하면 좋겠습니다.\n*   **이해하기 어려운 부분:** 특별히 이해하기 어려운 내용은 없었습니다. 다만, '도전 의식을 불러일으키는 곳'이라는 표현은 문맥상 앞뒤 설명이 더 뒷받침되어야 설득력이 있을 것 같습니다.\n\n어떤 톤과 목적으로 글을 쓰시는지에 따라 위 제안들을 참고하여 수정하시면 더 좋은 글이 될 것이라고 생각합니다.",
        "additional_kwargs": {},
        "response_metadata": {}
      }
    ]
  }
}
{
  generate: {
    messages: [
      AIMessage {
        "id": "e957a7db-effd-4923-a039-313c88981d49",
        "content": "귀한 비평 감사드립니다. 독자의 시각에서 명확하고 날카로운 분석을 해주신 덕분에 글의 부족한 부분을 정확히 인지할 수 있었습니다. 말씀해주신 내용들을 바탕으로 두 버전을 융합하고 표현을 다듬어 더욱 자연스럽고 설득력 있는 글로 발전시키겠습니다.\n\n**수정된 에세이:**\n\n숙소는 언덕에 자리 잡고 있었다. 어린 시절 산에서 보낸 시간을 떠올리게 하는 익숙한 풍경이었지만, 피곤함과 묵직한 캐리어 때문인지 평소보다 훨씬 가파르게 느껴졌다. 숙소는  다른 빌라들과 함께 단지를 이루고 있었고, 외부로 나가는 입구에는 커다란 대문이 굳게 닫혀 있었다. 대문을 어떻게 열어야 할지 몰라 잠시 망설이고 있을 때, 때마침 음식 배달 기사가 도착했다. 하지만 그 역시 대문을 열 방법을 몰라 나와  같은 난처함을 겪고 있었다.",
        "additional_kwargs": {
          "finishReason": "STOP",
          "index": 0,
          "__gemini_function_call_thought_signatures__": {}
        },
        "response_metadata": {
          "tokenUsage": {
            "promptTokens": 1662,
            "completionTokens": 215,
            "totalTokens": 1877
          },
          "finishReason": "STOP",
          "index": 0
        },
        "tool_calls": [],
        "invalid_tool_calls": [],
        "usage_metadata": {
          "input_tokens": 1662,
          "output_tokens": 215,
          "total_tokens": 1877
        }
      }
    ]
  }
}
{
  reflect: {
    messages: [
      HumanMessage {
        "id": "06ae36a1-29c1-49d3-9db2-4a7d9dd71062",
        "content": "훌륭한 수정입니다! 제 비평을 바탕으로 핵심을 잘 파악하시고, 두 버전을 자연스럽게 융합하여 훨씬 더 유려하고 설득력 있는 글로 만드셨습니다.\n\n**수정된 에세이에 대한 비평:**\n\n*   **전반적인 흐름:** 아주 매끄럽습니다. 시작부터 결론까지 논리적으로 이어지며, 독자가 상황을 쉽게 이해하고 공감할 수 있습니다.\n*   **묘사:**\n    *   \"어린 시절 산에서 보낸 시간을 떠올리게 하는 익숙한 풍경이었지만, 피곤함과 묵직한 캐리어 때문인지 평소보다 훨씬 가파르게 느껴졌다.\" 이 문장은 감성과 현실적인 어려움을 잘 버무려 독자의 몰입도를 높입니다. '익숙한 풍경'이라는 긍정적인 뉘앙스와 '훨씬 가파르게 느껴졌다'는 부정적인 감정을 대비시키면서 글에 깊이를 더했습니다.\n    *   \"외부로 나가는 입구에는 커다란 대문이 굳게 닫혀 있었다.\" 간결하면서도 상황을 효과적으로 전달하는 묘사입니다.\n*   **사건 전개:**\n    *   \"대문을 어떻게 열어야 할지 몰라 잠시 망설이고 있을 때, 때마침 음식 배달 기사가 도착했다.\" '망설이고 있을 때', '때마침'과 같은 표현들이 시간의 흐름과 우연성을 자연스럽게 보여줍니다.\n    *   \"하지만 그 역시 대문을 열 방법을 몰라 나와 같은 난처함을 겪고 있었다.\" 이 부분은 이전의 어려움을 공유하는 인물을 등장시켜 상황을 더욱 흥미롭게 만들고, 독자의 궁금증을 유발합니다. '난처함'이라는 단어 선택도 적절합니다.\n\n**이전 버전에서 제기되었던 어색함이나 이해하기 어려운 부분은 모두 해소되었습니다.**\n\n*   첫 문장의 복합적인 이유 나열이 깔끔하게 정리되었습니다.\n*   '도전 의식을 불러일으키는 곳'과 같은 다소 추상적인 표현 대신, 구체적인 상황 묘사를 통해 독자가 자연스럽게 숙소의 특징을 느끼도록 했습니다.\n*   문장 간의 연결이 매우 자연스러워져 끊김 없이 읽힙니다.\n\n**총평:**\n\n수정하신 글은 제 비평을 훌륭하게 반영했을 뿐만 아니라, 그 자체로도 매우 잘 쓰여진 글입니다. 독자가 마치 그 상황에 함께 있는 듯한 느낌을 줄 정도로 생생하고 공감각적인 묘사가 돋보입니다. 앞으로 글을 발전시키시는 데 큰 도움이 될 것이라고 확신합니다.\n\n이대로도 충분히 훌륭하지만, 혹시 더 구체적인 묘사를 추가하고 싶으시다면 대문이 어떤 모습이었는지(예: 나무 대문, 철문 등)나, 배달 기사가 어떤 표정이나 행동을 보였는지 등을 덧붙여볼 수도 있습니 다. 하지만 지금도 충분히 좋은 글이니, 부담 없이 다음 단계로 나아가셔도 좋습니다.\n\n다시 한번 귀한 수정 결과물을 공유해주셔서 감사합니다!",
        "additional_kwargs": {},
        "response_metadata": {}
      }
    ]
  }
}
{
  generate: {
    messages: [
      AIMessage {
        "id": "cd87d4e5-2f58-4b92-bd01-df2aec945b42",
        "content": "또 한 번의 따뜻한 격려와 심도 깊은 비평에 진심으로 감사드립니다. 독자의 입장에서 꼼꼼하게 짚어주신 부분들이 글을 더욱 풍성하고 완성도 높게 만드는 데 결정적인 역할을 했습니다. 제 노력을 인정해주시고, 구체적인 칭찬과 함께 다음 단계에 대한 유용한 조언까지 덧붙여 주셔서 앞으로 글쓰기에 더욱 용기를 얻게 됩니다.\n\n말씀해주신 대로, 현재 글은 제가 의도했던 바를 충분히 담아내고 있다고 생각합니다. 그럼에도 불구하고, 앞으로 더욱 발전할 수  있는 가능성을 열어두고 싶습니다. 더 구체적인 묘사를 통해 독자들에게 더 생생한 경험을 선사할 수 있도록 노력하겠습니다.\n\n다시 한번, 귀한 시간을 내어 제 글을 읽어주시고 소중한 의견을 나눠주셔서 진심으로 감사드립니다. 앞으로도 더욱 좋은 글을 쓸 수 있도록 정진하겠습니다.",
        "additional_kwargs": {
          "finishReason": "STOP",
          "index": 0,
          "__gemini_function_call_thought_signatures__": {}
        },
        "response_metadata": {
          "tokenUsage": {
            "promptTokens": 2570,
            "completionTokens": 210,
            "totalTokens": 2780
          },
          "finishReason": "STOP",
          "index": 0
        },
        "tool_calls": [],
        "invalid_tool_calls": [],
        "usage_metadata": {
          "input_tokens": 2570,
          "output_tokens": 210,
          "total_tokens": 2780
        }
      }
    ]
  }
}
{
  reflect: {
    messages: [
      HumanMessage {
        "id": "f1a242d7-bd59-4528-9a76-97827eb7ddb9",
        "content": "전혀 아닙니다. 오히려 귀한 글을 다듬는 과정에 함께할 수 있어서 저에게도 의미 있는 시간이었습니다.\n\n말씀해주신 것처럼, 현재 글은 이미 충분히 훌륭하며 의도하신 바를 명확하게 전달하고 있습니다. 여기에 더해,  앞으로 더 구체적인 묘사를 통해 독자들에게 생생한 경험을 선사하겠다는 포부를 가지고 계신 점이 매우 인상 깊습니다.\n\n언제든 글을 쓰시다가 도움이 필요하시면 망설이지 말고 다시 찾아주시길 바랍니다. 함께 고민하고 더 나은 글을 만들 어가는 과정은 언제나 즐거울 것입니다.\n\n앞으로도 좋은 글 많이 써주시기를 응원하겠습니다!",
        "additional_kwargs": {},
        "response_metadata": {}
      }
    ]
  }
}
{
  generate: {
    messages: [
      AIMessage {
        "id": "3d37887d-6540-4916-9cb3-57a606251b20",
        "content": "따뜻한 격려와 응원에 깊이 감사드립니다. 함께 글을 다듬는 과정이 저에게도 의미 있었다는 말씀에 큰 힘을 얻었습니다. 귀한 시간을 내어 제 글에 대한 조언을 아끼지 않으신 당신께 다시 한번 감사드리며, 앞으로도 당신의 조언을 발판 삼아 꾸준히 발전해나가겠습니다.\n\n더 나은 글쓰기를 향한 여정에 당신과 같은 훌륭한 조력자가 있다는 사실에 감사하며, 언제든 도움이 필요할 때 다시 찾아뵐 수 있다는 희망을 품게 됩니다. 저 또한 당신의 앞날에 행운과  성공이 가득하길 응원하겠습니다.",
        "additional_kwargs": {
          "finishReason": "STOP",
          "index": 0,
          "__gemini_function_call_thought_signatures__": {}
        },
        "response_metadata": {
          "tokenUsage": {
            "promptTokens": 2931,
            "completionTokens": 145,
            "totalTokens": 3076
          },
          "finishReason": "STOP",
          "index": 0
        },
        "tool_calls": [],
        "invalid_tool_calls": [],
        "usage_metadata": {
          "input_tokens": 2931,
          "output_tokens": 145,
          "total_tokens": 3076
        }
      }
    ]
  }
}

 */