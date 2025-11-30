import {z} from "zod";
import {ChatGoogleGenerativeAI} from "@langchain/google-genai";
import {AIMessage, HumanMessage, SystemMessage} from "@langchain/core/messages";
import {
  Annotation,
  END,
  messagesStateReducer,
  START,
  StateGraph
} from "@langchain/langgraph";

const SupervisorDecision = z.object({
  next: z.enum(['researcher', 'coder', 'FINISH'])
})

const model = new ChatGoogleGenerativeAI({model: "gemini-2.5-flash-lite"})
const annotation = Annotation.Root({
  messages: Annotation({reducer: messagesStateReducer, default: () => []}),
  next: Annotation({default: () => "FINISH"}),
})
const modelWithStructuredOutput = model.withStructuredOutput(SupervisorDecision)
const agents = ["researcher", "coder"]
const systemPrompt = `
  넌 서브 에이전트 사이의 대화를 관리하는 슈퍼바이저야.
  각 서브 에이전트는 임무를 수행하고 결과와 상태를 응답해.
  서브 에이전트 목록: ${agents.join(", ")}
  서브 에이전트 목록에서 다음으로 행동할 에이전트를 선택해, 작업이 완료된 경우에는 "FINISH"로 응답해.
`.trim().split('\n').map(s => s.trim()).join('\n')
const humanQuestion = `
  위 대화를 바탕으로 다음으로 행동할 서브 에이전트는 누구야? 아니면 작업이 완료되었어?
  반드시 다음 중 하나로 답해줘.
  ${agents.join(", ")}, FINISH
`.trim().split('\n').map(s => s.trim()).join('\n')

const supervisor = async (state) => {
  const messages = [
    new SystemMessage(systemPrompt),
    ...state.messages,
    new HumanMessage(humanQuestion),
  ]
  const result = await modelWithStructuredOutput.invoke(messages)
  return {messages: state.messages, next: result.next}
}

const researcherTem = ["const 대신 let을 이용하도록 수정해", "javascript에서 const는 상수로 선언과 동시에 값을 할당해야 해."]
const researcher = async (state) => {
  const response = new AIMessage(researcherTem.pop() ?? "더 이상 수정할 부분이 없어.")
  return {
    messages: [...state.messages, response]
  }
}
const coderTem = ["let some; if (a > 1) some = 1; else some = 10;", "const some; if (a > 1) some = 1; else some = 10;"]
const coder = async (state) => {
  const response = new AIMessage(coderTem.pop() ?? "코드 작성이 완료되었어.")
  return {
    messages: [...state.messages, response]
  }
}

const graph = new StateGraph(annotation)
.addNode("supervisor", supervisor)
.addNode("researcher", researcher)
.addNode("coder", coder)
.addEdge(START, "supervisor")
.addConditionalEdges("supervisor", async (state) => state.next === "FINISH" ? END : state.next)
.addEdge("researcher", "supervisor")
.addEdge("coder", "supervisor")
.compile()
const input = {
  messages: [
    new HumanMessage("javascript에서 const로 변수를 선언한 뒤 if-else문에서 값을 할당하려고 하는데 잘 안되어."),
  ],
}
for await (const chunk of await graph.stream(input)) {
  console.log(chunk)
}
