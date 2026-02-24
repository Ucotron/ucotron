'use client';

import { forwardRef } from 'react';
import { cn } from '../lib/utils';

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
  helperText?: string;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, label, error, helperText, id, ...props }, ref) => {
    const inputId = id || label?.toLowerCase().replace(/\s+/g, '-');

    return (
      <div className="w-full">
        {label && (
          <label htmlFor={inputId} className="block text-sm font-medium text-neutral-300 mb-1.5">
            {label}
          </label>
        )}
        <textarea
          ref={ref}
          id={inputId}
          className={cn(
            'w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white placeholder:text-neutral-500',
            'focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500/50',
            'transition-all duration-200 resize-none',
            error && 'border-red-500/50 focus:ring-red-500/50 focus:border-red-500/50',
            className
          )}
          {...props}
        />
        {(error || helperText) && (
          <p className={cn('mt-1.5 text-sm', error ? 'text-red-400' : 'text-neutral-500')}>
            {error || helperText}
          </p>
        )}
      </div>
    );
  }
);

Textarea.displayName = 'Textarea';
