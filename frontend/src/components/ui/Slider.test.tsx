import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Slider } from './slider'

describe('Slider Component', () => {
  it('should render correctly with default props', () => {
    render(<Slider defaultValue={[50]} max={100} min={0} step={1} />)
    const slider = screen.getByRole('slider')
    expect(slider).toBeInTheDocument()
  })

  it('should respect defaultValue prop', () => {
    render(<Slider defaultValue={[30]} max={100} min={0} step={1} />)
    const slider = screen.getByRole('slider')
    expect(slider).toHaveValue(30)
  })

  it('should be in the document when rendered', () => {
    const { container } = render(<Slider defaultValue={[50]} />)
    expect(container.firstChild).toBeInTheDocument()
  })

  it('should render without errors when provided with all props', () => {
    expect(() => {
      render(
        <Slider
          defaultValue={[25]}
          max={100}
          min={0}
          step={5}
          className="test-slider"
        />
      )
    }).not.toThrow()
  })

  it('should render without errors when disabled', () => {
    expect(() => {
      render(<Slider defaultValue={[50]} disabled />)
    }).not.toThrow()
  })
})
