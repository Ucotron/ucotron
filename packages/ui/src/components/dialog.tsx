'use client';

import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { cn } from '../lib/utils';

export interface DialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title?: string;
  description?: string;
  children?: React.ReactNode;
}

export function Dialog({ open, onOpenChange, title, description, children }: DialogProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (open) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [open]);

  if (!mounted) return null;

  return createPortal(
    <>
      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => onOpenChange(false)}
          />
          <div
            className={cn(
              'relative z-10 w-full max-w-lg mx-4 rounded-xl border border-white/10 bg-neutral-900/95 backdrop-blur shadow-xl',
              'animate-in fade-in zoom-in-95 duration-200'
            )}
          >
            <div className="flex items-start justify-between p-4 border-b border-white/10">
              <div>
                {title && <h2 className="text-lg font-semibold text-white">{title}</h2>}
                {description && <p className="text-sm text-neutral-400 mt-1">{description}</p>}
              </div>
              <button
                onClick={() => onOpenChange(false)}
                className="p-1 rounded-lg text-neutral-400 hover:text-white hover:bg-white/10 transition-colors"
              >
                <X size={20} />
              </button>
            </div>
            <div className="p-4">{children}</div>
          </div>
        </div>
      )}
    </>,
    document.body
  );
}
