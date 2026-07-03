// Undo/redo для редактора масок. Упрощение прототипа: стек хранит полные снапшоты
// Uint8Array (working-разрешение ≤1536px, это единицы МБ), а не dirty-rect патчи —
// так надёжнее и на порядок проще при рисовании произвольными инструментами.

import { useRef, useState, useCallback } from 'react';

const HISTORY_LIMIT = 20;

export function useUndoRedo() {
  const undoStack = useRef<Uint8Array[]>([]);
  const redoStack = useRef<Uint8Array[]>([]);
  const [, forceRender] = useState(0);
  const bump = () => forceRender((n) => n + 1);

  const snapshotBeforeEdit = useCallback((current: Uint8Array) => {
    undoStack.current.push(current.slice());
    if (undoStack.current.length > HISTORY_LIMIT) undoStack.current.shift();
    redoStack.current = [];
    bump();
  }, []);

  const undo = useCallback((current: Uint8Array): Uint8Array | null => {
    const prev = undoStack.current.pop();
    if (!prev) return null;
    redoStack.current.push(current.slice());
    bump();
    return prev;
  }, []);

  const redo = useCallback((current: Uint8Array): Uint8Array | null => {
    const next = redoStack.current.pop();
    if (!next) return null;
    undoStack.current.push(current.slice());
    bump();
    return next;
  }, []);

  const clear = useCallback(() => {
    undoStack.current = [];
    redoStack.current = [];
    bump();
  }, []);

  return {
    snapshotBeforeEdit,
    undo,
    redo,
    clear,
    canUndo: undoStack.current.length > 0,
    canRedo: redoStack.current.length > 0,
  };
}
