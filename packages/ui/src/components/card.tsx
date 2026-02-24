'use client';

import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../lib/utils';
import type { ReactNode } from 'react';

const cardVariants = cva(
  'rounded-xl border backdrop-blur transition-all duration-200',
  {
    variants: {
      variant: {
        default: 'bg-white/5 border-white/10',
        elevated: 'bg-white/5 border-white/10 shadow-lg',
        interactive: 'bg-white/5 border-white/10 hover:scale-[1.02] hover:shadow-lg cursor-pointer',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
);

export interface CardProps extends VariantProps<typeof cardVariants> {
  title?: string;
  description?: string;
  children?: ReactNode;
  footer?: ReactNode;
  className?: string;
}

export function Card({ title, description, children, footer, variant, className }: CardProps) {
  return (
    <div className={cn(cardVariants({ variant, className }))}>
      {(title || description) && (
        <div className="p-4 border-b border-white/10">
          {title && <h3 className="text-lg font-semibold text-white">{title}</h3>}
          {description && <p className="text-sm text-neutral-400 mt-1">{description}</p>}
        </div>
      )}
      {children && <div className="p-4">{children}</div>}
      {footer && <div className="p-4 border-t border-white/10">{footer}</div>}
    </div>
  );
}
