/**
 * Table组件
 */

import React from 'react';

interface TableProps extends React.TableHTMLAttributes<HTMLTableElement> {
  children: React.ReactNode;
}

interface TableHeaderProps extends React.HTMLAttributes<HTMLTableSectionElement> {
  children: React.ReactNode;
}

interface TableBodyProps extends React.HTMLAttributes<HTMLTableSectionElement> {
  children: React.ReactNode;
}

interface TableRowProps extends React.HTMLAttributes<HTMLTableRowElement> {
  children: React.ReactNode;
  isStriped?: boolean;
}

interface TableHeadProps extends React.ThHTMLAttributes<HTMLTableCellElement> {
  children: React.ReactNode;
  sortable?: boolean;
  sorted?: 'asc' | 'desc' | false;
  onSort?: () => void;
}

interface TableCellProps extends React.TdHTMLAttributes<HTMLTableCellElement> {
  children: React.ReactNode;
}

// 表格容器
export const Table = React.forwardRef<HTMLTableElement, TableProps>(
  ({ className, children, ...props }, ref) => (
      <div className="relative w-full overflow-auto">
        <table
          ref={ref}
          className={`w-full border-collapse text-left ${className}`}
          {...props}
        >
          {children}
        </table>
      </div>
    ),
);
Table.displayName = 'Table';

// 表格头部
export const TableHeader = React.forwardRef<HTMLTableSectionElement, TableHeaderProps>(
  ({ className, children, ...props }, ref) => (
      <thead
        ref={ref}
        className={`border-b border-tech-light/50 ${className}`}
        {...props}
      >
        {children}
      </thead>
    ),
);
TableHeader.displayName = 'TableHeader';

// 表格主体
export const TableBody = React.forwardRef<HTMLTableSectionElement, TableBodyProps>(
  ({ className, children, ...props }, ref) => (
      <tbody
        ref={ref}
        className={className}
        {...props}
      >
        {children}
      </tbody>
    ),
);
TableBody.displayName = 'TableBody';

// 表格行
export const TableRow = React.forwardRef<HTMLTableRowElement, TableRowProps>(
  ({ className, children, isStriped = false, ...props }, ref) => (
      <tr
        ref={ref}
        className={`border-b border-tech-light/30 hover:bg-tech-light/10 transition-colors ${isStriped ? 'bg-tech-dark/20' : ''} ${className}`}
        {...props}
      >
        {children}
      </tr>
    ),
);
TableRow.displayName = 'TableRow';

// 表格表头单元格
export const TableHead = React.forwardRef<HTMLTableCellElement, TableHeadProps>(
  ({ className, children, sortable = false, sorted = false, onSort, ...props }, ref) => (
      <th
        ref={ref}
        className={`h-12 px-6 py-4 font-medium text-sm text-gray-300 ${sortable ? 'cursor-pointer hover:text-tech-primary transition-colors' : 'text-gray-400'} ${className}`}
        {...props}
      >
        <div className="flex items-center space-x-2">
          {children}
          {sortable && (
            <span className={`text-xs ${sorted ? 'text-tech-primary' : 'text-gray-500'}`}>
              {sorted === 'asc' ? '↑' : sorted === 'desc' ? '↓' : '↕'}
            </span>
          )}
        </div>
      </th>
    ),
);
TableHead.displayName = 'TableHead';

// 表格数据单元格
export const TableCell = React.forwardRef<HTMLTableCellElement, TableCellProps>(
  ({ className, children, ...props }, ref) => (
      <td
        ref={ref}
        className={`p-6 text-sm text-gray-200 ${className}`}
        {...props}
      >
        {children}
      </td>
    ),
);
TableCell.displayName = 'TableCell';

// 分页组件
export interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  className?: string;
}

export const Pagination = ({ currentPage, totalPages, onPageChange, className = '' }: PaginationProps) => {
  const pages = Array.from({ length: totalPages }, (_, i) => i + 1);

  return (
    <div className={`flex items-center justify-between mt-6 px-4 ${className}`}>
      <div className="text-sm text-gray-400">
        第 {currentPage} / {totalPages} 页
      </div>
      <div className="flex items-center space-x-2">
        <button
          onClick={() => onPageChange(Math.max(1, currentPage - 1))}
          disabled={currentPage === 1}
          className="px-3 py-1 rounded-md border border-tech-light/50 bg-tech-dark hover:bg-tech-light/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          上一页
        </button>

        {pages.map(page => (
          <button
            key={page}
            onClick={() => onPageChange(page)}
            className={`px-3 py-1 rounded-md transition-colors ${currentPage === page ?
              'bg-tech-primary/30 border border-tech-primary text-tech-primary' :
              'border border-tech-light/50 bg-tech-dark hover:bg-tech-light/20'}`}
          >
            {page}
          </button>
        ))}

        <button
          onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
          disabled={currentPage === totalPages}
          className="px-3 py-1 rounded-md border border-tech-light/50 bg-tech-dark hover:bg-tech-light/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          下一页
        </button>
      </div>
    </div>
  );
};
