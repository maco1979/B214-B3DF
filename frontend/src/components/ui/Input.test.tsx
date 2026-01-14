import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Input } from './input';

describe('Input Component', () => {
  it('should render correctly with default props', () => {
    render(<Input placeholder="Enter text" />);
    const input = screen.getByPlaceholderText('Enter text');
    expect(input).toBeInTheDocument();
    expect(input).toHaveClass('flex');
    expect(input).toHaveClass('h-10');
    expect(input).toHaveClass('w-full');
  });

  it('should display the correct value when provided', () => {
    render(<Input value="test value" />);
    const input = screen.getByDisplayValue('test value');
    expect(input).toBeInTheDocument();
    expect(input).toHaveValue('test value');
  });

  it('should handle input changes', () => {
    const handleChange = vi.fn();
    render(<Input onChange={handleChange} placeholder="Enter text" />);
    const input = screen.getByPlaceholderText('Enter text');

    fireEvent.change(input, { target: { value: 'new value' } });
    expect(handleChange).toHaveBeenCalledTimes(1);
  });

  it('should support placeholder text', () => {
    render(<Input placeholder="Custom placeholder" />);
    const input = screen.getByPlaceholderText('Custom placeholder');
    expect(input).toHaveAttribute('placeholder', 'Custom placeholder');
  });

  it('should be disabled when disabled prop is true', () => {
    render(<Input disabled value="disabled value" />);
    const input = screen.getByDisplayValue('disabled value');
    expect(input).toBeDisabled();
    expect(input).toHaveClass('disabled:cursor-not-allowed');
  });

  it('should render with custom className', () => {
    render(<Input className="custom-input" placeholder="Custom class" />);
    const input = screen.getByPlaceholderText('Custom class');
    expect(input).toHaveClass('custom-input');
  });

  it('should support different input types', () => {
    render(<Input type="password" placeholder="Password" />);
    const input = screen.getByPlaceholderText('Password');
    expect(input).toHaveAttribute('type', 'password');
  });
});
