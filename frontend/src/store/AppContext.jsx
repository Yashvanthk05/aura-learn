import { createContext, useContext, useReducer } from "react";

const AppContext = createContext(null);

const initialState = {
  user: null,
  chats: [],
  activeChatId: null,
  activeChat: null,
  sources: [],

  isCreatingChat: false,
  createChatError: null,

  activeDocumentId: null,
  isUploading: false,
  uploadError: null,

  sessionId: null,
  messages: [],
  isChatLoading: false,
  chatError: null,

  activeFeature: null,
  featureResult: null,
  isFeatureLoading: false,
  featureError: null,
};

function reducer(state, action) {
  switch (action.type) {
    case "SET_USER":
      return { ...state, user: action.payload };

    case "LOGOUT":
      return {
        ...state,
        user: null,
        chats: [],
        activeChatId: null,
        activeChat: null,
        sources: [],
        activeDocumentId: null,
        sessionId: null,
        messages: [],
        featureResult: null,
        createChatError: null,
      };

    case "SET_CHAT_SESSIONS":
      return { ...state, chats: action.payload };

    case "ADD_CHAT_SESSION":
      return {
        ...state,
        chats: [action.payload, ...state.chats.filter((c) => c.session_id !== action.payload.session_id)],
      };

    case "UPDATE_CHAT_SESSION": {
      const { sessionId, patch } = action.payload;
      const chats = state.chats.map((chat) =>
        chat.session_id === sessionId ? { ...chat, ...patch } : chat,
      );

      const isActive = state.activeChat?.session_id === sessionId;
      return {
        ...state,
        chats,
        activeChat: isActive ? { ...state.activeChat, ...patch } : state.activeChat,
      };
    }

    case "SET_ACTIVE_CHAT": {
      if (!action.payload) {
        return {
          ...state,
          activeChatId: null,
          activeChat: null,
          sessionId: null,
          activeDocumentId: null,
          messages: [],
          sources: [],
          featureResult: null,
          featureError: null,
        };
      }

      return {
        ...state,
        activeChatId: action.payload.session_id,
        activeChat: action.payload,
        sessionId: action.payload.session_id,
        activeDocumentId: action.payload.document_id,
        messages: [],
        featureResult: null,
        featureError: null,
      };
    }

    case "SET_SOURCES":
      return { ...state, sources: action.payload };

    case "ADD_SOURCE":
      return { ...state, sources: [...state.sources, action.payload] };

    case "SET_CREATING_CHAT":
      return {
        ...state,
        isCreatingChat: action.payload,
        createChatError: action.payload ? null : state.createChatError,
      };

    case "SET_CREATE_CHAT_ERROR":
      return {
        ...state,
        isCreatingChat: false,
        createChatError: action.payload,
      };

    case "SET_UPLOADING":
      return { ...state, isUploading: action.payload, uploadError: null };

    case "SET_UPLOAD_ERROR":
      return { ...state, isUploading: false, uploadError: action.payload };

    case "SET_SESSION":
      return { ...state, sessionId: action.payload };

    case "ADD_MESSAGE":
      return { ...state, messages: [...state.messages, action.payload] };

    case "SET_MESSAGES":
      return { ...state, messages: action.payload };

    case "SET_CHAT_LOADING":
      return { ...state, isChatLoading: action.payload, chatError: null };

    case "SET_CHAT_ERROR":
      return { ...state, isChatLoading: false, chatError: action.payload };

    case "SET_ACTIVE_FEATURE":
      return {
        ...state,
        activeFeature: action.payload,
        featureResult: null,
        featureError: null,
      };

    case "SET_FEATURE_RESULT":
      return {
        ...state,
        featureResult: action.payload,
        isFeatureLoading: false,
        featureError: null,
      };

    case "SET_FEATURE_LOADING":
      return {
        ...state,
        isFeatureLoading: action.payload,
        featureError: null,
      };

    case "SET_FEATURE_ERROR":
      return {
        ...state,
        isFeatureLoading: false,
        featureError: action.payload,
      };

    default:
      return state;
  }
}

export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}
