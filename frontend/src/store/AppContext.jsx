import { createContext, useContext, useReducer } from "react";

const AppContext = createContext(null);

const initialState = {
  documents: [],
  activeDocumentId: null,
  activeDocument: null,
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
    case "SET_DOCUMENTS":
      return { ...state, documents: action.payload };

    case "ADD_DOCUMENT":
      return {
        ...state,
        documents: [...state.documents, action.payload],
        isUploading: false,
        uploadError: null,
      };

    case "REMOVE_DOCUMENT":
      return {
        ...state,
        documents: state.documents.filter(
          (d) => d.document_id !== action.payload
        ),
        activeDocumentId:
          state.activeDocumentId === action.payload
            ? null
            : state.activeDocumentId,
        activeDocument:
          state.activeDocumentId === action.payload
            ? null
            : state.activeDocument,
      };

    case "SET_ACTIVE_DOCUMENT":
      return {
        ...state,
        activeDocumentId: action.payload.id,
        activeDocument: action.payload.data,
        sessionId: null,
        messages: [],
        featureResult: null,
        featureError: null,
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
