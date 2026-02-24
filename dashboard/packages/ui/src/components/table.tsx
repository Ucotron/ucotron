'use client';

import { cn } from '../lib/utils';
import type { ReactNode } from 'react';

export interface TableColumn<T = unknown> {
  key: string;
  header: string;
  render?: (row: T) => ReactNode;
  sortable?: boolean;
}

export interface TableProps<T = unknown> {
  columns: TableColumn<T>[];
  data: T[];
  sortable?: boolean;
  onSort?: (key: string, direction: 'asc' | 'desc') => void;
  className?: string;
}

export function Table<T extends Record<string, unknown>>({ columns, data, sortable, onSort, className }: TableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const handleSort = (key: string) => {
    if (!sortable || !onSort) return;
    const newDirection = sortKey === key && sortDirection === 'asc' ? 'desc' : 'asc';
    setSortKey(key);
    setSortDirection(newDirection);
    onSort(key, newDirection);
  };

  return (
    <div className={cn('w-full overflow-x-auto rounded-lg border border-white/10', className)}>
      <table className="w-full">
        <thead>
          <tr className="border-b border-white/10 bg-white/5">
            {columns.map((col) => (
              <th
                key={col.key}
                className={cn(
                  'px-4 py-3 text-left text-sm font-medium text-neutral-300',
                  sortable && col.sortable && 'cursor-pointer hover:text-white transition-colors'
                )}
                onClick={() => sortable && col.sortable && handleSort(col.key)}
              >
                <div className="flex items-center gap-1">
                  {col.header}
                  {sortable && col.sortable && sortKey === col.key && (
                    <span className="text-primary-400">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="px-4 py-8 text-center text-neutral-500">
                No data available
              </td>
            </tr>
          ) : (
            data.map((row, i) => (
              <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                {columns.map((col) => (
                  <td key={col.key} className="px-4 py-3 text-sm text-neutral-300">
                    {col.render ? col.render(row) : (row[col.key] as ReactNode)}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

import { useState } from 'react';

export const TableHeader = Table;
export const TableBody = ({ children, className }: { children: ReactNode; className?: string }) => (
  <tbody className={cn(className)}>{children}</tbody>
);
export const TableRow = ({ children, className }: { children: ReactNode; className?: string }) => (
  <tr className={cn('border-b border-white/5', className)}>{children}</tr>
);
export const TableCell = ({ children, className }: { children: ReactNode; className?: string }) => (
  <td className={cn('px-4 py-3 text-sm text-neutral-300', className)}>{children}</td>
);
