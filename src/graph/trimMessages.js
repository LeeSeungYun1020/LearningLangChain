import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  trimMessages
} from "@langchain/core/messages";
import {ChatGoogleGenerativeAI} from "@langchain/google-genai";

const messages = [
  new SystemMessage("당신은 마침표 대신 느낌표로 대답하는 어시스턴트예요."),
  new HumanMessage("안녕하세요."),
  new AIMessage("안녕하세요! 궁금한 질문을 입력해주시면 제가 답변해 드릴게요!"),
  new HumanMessage("제가 궁금한 질문을 그냥 입력하면 되나요?"),
  new AIMessage("네 궁금하신 질문을 입력해 주세요!"),
  new HumanMessage("질문 형식에 어떠한 제한이 있나요?"),
  new AIMessage("아니요! 없습니다!"),
  new HumanMessage("카카오 가격은 얼마입니까?"),
  new AIMessage("1Kg 당 5달러 수준입니다!"),
  new HumanMessage("오렌지 주스 가격은 얼마입니까?"),
  new AIMessage("오렌지 주스 가격은 0.45Kg 당 138 달러 수준입니다!"),
  new HumanMessage("원두 가격은 얼마입니까?"),
  new AIMessage("원두 가격은 1Kg 당 0.45달러 수준입니다!"),
]

const trimmer = trimMessages({
  maxTokens: 300,
  strategy: "last",
  tokenCounter: new ChatGoogleGenerativeAI({model: "gemini-2.5-flash-lite"}),
  includeSystem: true,
  allowPartial: false,
  startOn: 'human',
})

console.log(await trimmer.invoke(messages))
// 카카오 가격 질문 이전에 전송한 메시지가 삭제됨(includeSystem 이므로 시스템 메시지는 제외)