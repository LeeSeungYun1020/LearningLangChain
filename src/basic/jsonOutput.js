import {initChatModel} from "langchain";
import {z} from "zod"

const answerSchema = z.object({
  answer: z.string().describe("사용자 질문에 따른 답변"),
  justification: z.string().describe("답변 근거"),
}).describe("사용자 질문에 대한 답변과 근거를 제공해")

let model = await initChatModel("google-genai:gemini-2.5-flash-lite")
model = model.withStructuredOutput(answerSchema)

const response = await model.invoke("1kg 깃털과 1kg 돌 중 어떤 것이 더 무거운가?")
console.log(response)
// {
//   answer: '1kg 깃털과 1kg 돌은 무게가 같습니다.',
//   justification: '둘 다 1kg으로 동일한 질량을 가지고 있기 때문에 무게도 같습니다. 무게는 질량에 중력 가속도를 곱한 값인데, 질량이 같으면 중력 가속도도 같으므로 무게도 같습니다.'
// }