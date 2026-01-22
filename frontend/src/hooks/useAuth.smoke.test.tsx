import { describe, it, expect } from 'vitest';

console.log('loaded smoke test');

describe('smoke', () => {
  console.log('inside describe smoke');
  it('works', () => {
    expect(true).toBe(true);
  });
});
