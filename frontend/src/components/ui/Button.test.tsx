import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Button } from './button'

describe('Button Component', () => {
  it('should render correctly with default variant', () => {
    render(<Button>Default Button</Button>)
    const button = screen.getByText('Default Button')
    expect(button).toBeInTheDocument()
    expect(button).toHaveClass('bg-primary')
  })

  it('should render correctly with different variants', () => {
    const variants = ['default', 'destructive', 'outline', 'secondary', 'ghost', 'link', 'tech']
    
    variants.forEach(variant => {
      render(<Button variant={variant as any}>{variant} Button</Button>)
      const button = screen.getByText(`${variant} Button`)
      expect(button).toBeInTheDocument()
    })
  })

  it('should render correctly with different sizes', () => {
    const sizes = ['default', 'sm', 'lg', 'icon']
    
    sizes.forEach(size => {
      render(<Button size={size as any}>{size} Button</Button>)
      const button = screen.getByText(`${size} Button`)
      expect(button).toBeInTheDocument()
    })
  })

  it('should handle click events', () => {
    const handleClick = vi.fn()
    render(<Button onClick={handleClick}>Clickable Button</Button>)
    const button = screen.getByText('Clickable Button')
    
    fireEvent.click(button)
    expect(handleClick).toHaveBeenCalledTimes(1)
  })

  it('should be disabled when disabled prop is true', () => {
    const handleClick = vi.fn()
    render(<Button disabled onClick={handleClick}>Disabled Button</Button>)
    const button = screen.getByText('Disabled Button')
    
    expect(button).toBeDisabled()
    fireEvent.click(button)
    expect(handleClick).not.toHaveBeenCalled()
  })

  it('should render with custom className', () => {
    render(<Button className="custom-class">Custom Button</Button>)
    const button = screen.getByText('Custom Button')
    
    expect(button).toHaveClass('custom-class')
  })
})
