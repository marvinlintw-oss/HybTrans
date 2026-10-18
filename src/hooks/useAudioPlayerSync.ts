import { useRef, useCallback } from 'react';
import { useAppStore } from '../store/useAppStore';

export function useAudioPlayerSync() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const setCurrentTime = useAppStore((s) => s.setCurrentTime);

  const seekToWord = useCallback((startSec: number, endSec?: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = startSec;
    audioRef.current.play();

    if (endSec) {
      const stopCheck = () => {
        if (audioRef.current && audioRef.current.currentTime >= endSec) {
          audioRef.current.pause();
          audioRef.current.removeEventListener('timeupdate', stopCheck);
        }
      };
      audioRef.current.addEventListener('timeupdate', stopCheck);
    }
  }, []);

  const handleTimeUpdate = useCallback(() => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  }, [setCurrentTime]);

  return { audioRef, seekToWord, handleTimeUpdate };
}