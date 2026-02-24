'use client';

import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Check, Search } from 'lucide-react';
import { cn } from '../lib/utils';

export interface SelectOption {
  value: string;
  label: string;
}

export interface SelectProps {
  options: SelectOption[];
  value?: string | string[];
  onChange: (value: string | string[]) => void;
  placeholder?: string;
  multiple?: boolean;
  searchable?: boolean;
  className?: string;
}

export function Select({
  options,
  value,
  onChange,
  placeholder = 'Select...',
  multiple = false,
  searchable = false,
  className,
}: SelectProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const ref = useRef<HTMLDivElement>(null);

  const selectedValues = Array.isArray(value) ? value : value ? [value] : [];
  const filteredOptions = searchable
    ? options.filter((opt) => opt.label.toLowerCase().includes(search.toLowerCase()))
    : options;

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        setOpen(false);
        setSearch('');
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (optionValue: string) => {
    if (multiple) {
      const newValue = selectedValues.includes(optionValue)
        ? selectedValues.filter((v) => v !== optionValue)
        : [...selectedValues, optionValue];
      onChange(newValue);
    } else {
      onChange(optionValue);
      setOpen(false);
      setSearch('');
    }
  };

  const displayValue = () => {
    if (selectedValues.length === 0) return placeholder;
    if (multiple) {
      const labels = options.filter((o) => selectedValues.includes(o.value)).map((o) => o.label);
      return labels.length > 2 ? `${labels.length} selected` : labels.join(', ');
    }
    return options.find((o) => o.value === value)?.label || placeholder;
  };

  return (
    <div ref={ref} className={cn('relative w-full', className)}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className={cn(
          'w-full h-10 px-3 rounded-lg bg-white/5 border border-white/10 text-left flex items-center justify-between',
          'focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500/50',
          'transition-all duration-200'
        )}
      >
        <span className={cn('truncate', selectedValues.length === 0 && 'text-neutral-500')}>
          {displayValue()}
        </span>
        <ChevronDown size={16} className={cn('text-neutral-400 transition-transform', open && 'rotate-180')} />
      </button>

      {open && (
        <div className="absolute z-50 w-full mt-1 py-1 rounded-lg bg-neutral-900 border border-white/10 shadow-xl">
          {searchable && (
            <div className="px-2 pb-2 border-b border-white/10">
              <div className="relative">
                <Search size={14} className="absolute left-2 top-1/2 -translate-y-1/2 text-neutral-500" />
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search..."
                  className="w-full h-8 pl-7 pr-2 rounded bg-white/5 border border-white/10 text-sm text-white placeholder:text-neutral-500 focus:outline-none focus:ring-1 focus:ring-primary-500/50"
                />
              </div>
            </div>
          )}
          <div className="max-h-60 overflow-y-auto py-1">
            {filteredOptions.length === 0 ? (
              <div className="px-3 py-2 text-sm text-neutral-500">No options found</div>
            ) : (
              filteredOptions.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => handleSelect(option.value)}
                  className={cn(
                    'w-full px-3 py-2 text-left text-sm flex items-center justify-between',
                    'hover:bg-white/10 transition-colors',
                    selectedValues.includes(option.value) ? 'text-white' : 'text-neutral-300'
                  )}
                >
                  <span className="truncate">{option.label}</span>
                  {selectedValues.includes(option.value) && <Check size={14} />}
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
