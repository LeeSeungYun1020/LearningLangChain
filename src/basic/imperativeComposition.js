import {initChatModel} from "langchain"
import {ChatPromptTemplate} from "@langchain/core/prompts";
import {RunnableLambda} from "@langchain/core/runnables";

const template = ChatPromptTemplate.fromMessages([
  ["system", "마침표 대신 느낌표 2개를 붙이는 어시스턴트입니다."],
  ["human", "{question}"],
])
const model = await initChatModel("google-genai:gemini-2.5-flash-lite")

const chatbot = RunnableLambda.from(async (values) => {
  const prompt = await template.invoke(values)
  return await model.invoke(prompt)
})
const response = await chatbot.invoke({
  question: "너의 이름은 뭐야?",
})

console.log(response)

// AIMessage {
//   "content": "저는 Google에서 훈련한 대규모 언어 모델입니다!!",
//   "additional_kwargs": {
//     "finishReason": "STOP",
//     "index": 0,
//     "__gemini_function_call_thought_signatures__": {}
//   },
//   "response_metadata": {
//     "tokenUsage": {
//       "promptTokens": 26,
//       "completionTokens": 13,
//       "totalTokens": 39
//     },
//     "finishReason": "STOP",
//     "index": 0
//   },
//   "tool_calls": [],
//   "invalid_tool_calls": [],
//   "usage_metadata": {
//     "input_tokens": 26,
//     "output_tokens": 13,
//     "total_tokens": 39
//   }
// }