import { ResponseProcessor } from './ResponseProcessor';
import { HttpResponse } from '../../types';

describe('ResponseProcessor', () => {
  let responseProcessor: ResponseProcessor;

  beforeEach(() => {
    responseProcessor = new ResponseProcessor();
  });

  describe('parseJsonResponse', () => {
    it('should parse JSON string response', () => {
      const response: HttpResponse = {
        status: 200,
        statusText: 'OK',
        headers: { 'Content-Type': 'application/json' },
        data: '{"success": true, "message": "Test response"}',
        duration: 100,
      };

      const result = responseProcessor.parseJsonResponse(response);

      expect(result).toEqual({ success: true, message: 'Test response' });
    });

    it('should return already parsed JSON data', () => {
      const response: HttpResponse = {
        status: 200,
        statusText: 'OK',
        headers: { 'Content-Type': 'application/json' },
        data: { success: true, message: 'Test response' },
        duration: 100,
      };

      const result = responseProcessor.parseJsonResponse(response);

      expect(result).toEqual({ success: true, message: 'Test response' });
    });

    it('should throw error for invalid JSON', () => {
      const response: HttpResponse = {
        status: 200,
        statusText: 'OK',
        headers: { 'Content-Type': 'application/json' },
        data: '{"success": true, "message": "Test response"', // Invalid JSON (missing closing brace)
        duration: 100,
      };

      expect(() => responseProcessor.parseJsonResponse(response)).toThrow('Failed to parse JSON response');
    });
  });

  describe('extractDataWithJsonPath', () => {
    it('should extract data with JSONPath', () => {
      const data = {
        users: [
          { id: 1, name: 'John', email: 'john@example.com' },
          { id: 2, name: 'Jane', email: 'jane@example.com' },
        ],
        total: 2,
        success: true,
      };

      const result = responseProcessor.extractDataWithJsonPath(data, '$.users[*].name');

      expect(result).toEqual(['John', 'Jane']);
    });

    it('should return single value for single match', () => {
      const data = {
        success: true,
        message: 'Test response',
        data: { id: 1, name: 'Test' },
      };

      const result = responseProcessor.extractDataWithJsonPath(data, '$.message');

      expect(result).toBe('Test response');
    });

    it('should return empty array for no matches', () => {
      const data = {
        success: true,
        message: 'Test response',
      };

      const result = responseProcessor.extractDataWithJsonPath(data, '$.nonexistent');

      expect(result).toEqual([]);
    });
  });

  describe('transformResponse', () => {
    it('should transform response using JSONPath mappings', () => {
      const response: HttpResponse = {
        status: 200,
        statusText: 'OK',
        headers: { 'Content-Type': 'application/json' },
        data: {
          users: [
            { id: 1, name: 'John', email: 'john@example.com' },
            { id: 2, name: 'Jane', email: 'jane@example.com' },
          ],
          total: 2,
          success: true,
        },
        duration: 100,
      };

      const transformConfig = {
        type: 'jsonPath' as const,
        mappings: {
          userName: '$.users[0].name',
          userEmail: '$.users[0].email',
          totalUsers: '$.total',
          allUserNames: '$.users[*].name',
        },
      };

      const result = responseProcessor.transformResponse(response, transformConfig);

      expect(result).toEqual({
        userName: 'John',
        userEmail: 'john@example.com',
        totalUsers: 2,
        allUserNames: ['John', 'Jane'],
      });
    });
  });

  describe('validateResponse', () => {
    it('should validate response status code', () => {
      const response: HttpResponse = {
        status: 200,
        statusText: 'OK',
        headers: { 'Content-Type': 'application/json' },
        data: { success: true },
        duration: 100,
      };

      const validationConfig = {
        statusCode: 200,
      };

      const result = responseProcessor.validateResponse(response, validationConfig);

      expect(result).toBe(true);
    });

    it('should validate response against JSON schema', () => {
      const response: HttpResponse = {
        status: 200,
        statusText: 'OK',
        headers: { 'Content-Type': 'application/json' },
        data: {
          id: 1,
          name: 'Test',
          email: 'test@example.com',
          active: true,
        },
        duration: 100,
      };

      const validationConfig = {
        statusCode: 200,
        jsonSchema: {
          type: 'object',
          required: ['id', 'name', 'email'],
          properties: {
            id: { type: 'number' },
            name: { type: 'string' },
            email: { type: 'string' },
            active: { type: 'boolean' },
          },
        },
      };

      const result = responseProcessor.validateResponse(response, validationConfig);

      expect(result).toBe(true);
    });

    it('should throw error for invalid status code', () => {
      const response: HttpResponse = {
        status: 404,
        statusText: 'Not Found',
        headers: { 'Content-Type': 'application/json' },
        data: { error: 'Resource not found' },
        duration: 100,
      };

      const validationConfig = {
        statusCode: 200,
      };

      expect(() => responseProcessor.validateResponse(response, validationConfig)).toThrow('Status code validation failed');
    });

    it('should throw error for missing required field in JSON schema', () => {
      const response: HttpResponse = {
        status: 200,
        statusText: 'OK',
        headers: { 'Content-Type': 'application/json' },
        data: {
          id: 1,
          name: 'Test',
          // Missing email field
          active: true,
        },
        duration: 100,
      };

      const validationConfig = {
        statusCode: 200,
        jsonSchema: {
          type: 'object',
          required: ['id', 'name', 'email'],
          properties: {
            id: { type: 'number' },
            name: { type: 'string' },
            email: { type: 'string' },
            active: { type: 'boolean' },
          },
        },
      };

      expect(() => responseProcessor.validateResponse(response, validationConfig)).toThrow('Missing required field: email');
    });
  });

  describe('isSuccessResponse', () => {
    it('should return true for 2xx status codes', () => {
      const response200: HttpResponse = {
        status: 200,
        statusText: 'OK',
        headers: {},
        data: {},
        duration: 100,
      };

      const response201: HttpResponse = {
        status: 201,
        statusText: 'Created',
        headers: {},
        data: {},
        duration: 100,
      };

      const response204: HttpResponse = {
        status: 204,
        statusText: 'No Content',
        headers: {},
        data: {},
        duration: 100,
      };

      expect(responseProcessor.isSuccessResponse(response200)).toBe(true);
      expect(responseProcessor.isSuccessResponse(response201)).toBe(true);
      expect(responseProcessor.isSuccessResponse(response204)).toBe(true);
    });

    it('should return false for non-2xx status codes', () => {
      const response400: HttpResponse = {
        status: 400,
        statusText: 'Bad Request',
        headers: {},
        data: {},
        duration: 100,
      };

      const response401: HttpResponse = {
        status: 401,
        statusText: 'Unauthorized',
        headers: {},
        data: {},
        duration: 100,
      };

      const response500: HttpResponse = {
        status: 500,
        statusText: 'Internal Server Error',
        headers: {},
        data: {},
        duration: 100,
      };

      expect(responseProcessor.isSuccessResponse(response400)).toBe(false);
      expect(responseProcessor.isSuccessResponse(response401)).toBe(false);
      expect(responseProcessor.isSuccessResponse(response500)).toBe(false);
    });
  });

  describe('getErrorMessage', () => {
    it('should get error message from response data with error field', () => {
      const response: HttpResponse = {
        status: 400,
        statusText: 'Bad Request',
        headers: { 'Content-Type': 'application/json' },
        data: { error: 'Invalid request parameters' },
        duration: 100,
      };

      const result = responseProcessor.getErrorMessage(response);

      expect(result).toBe('Invalid request parameters');
    });

    it('should get error message from response data with message field', () => {
      const response: HttpResponse = {
        status: 404,
        statusText: 'Not Found',
        headers: { 'Content-Type': 'application/json' },
        data: { message: 'Resource not found' },
        duration: 100,
      };

      const result = responseProcessor.getErrorMessage(response);

      expect(result).toBe('Resource not found');
    });

    it('should get error message from response data with errors array', () => {
      const response: HttpResponse = {
        status: 400,
        statusText: 'Bad Request',
        headers: { 'Content-Type': 'application/json' },
        data: {
          errors: [
            'Field "name" is required',
            'Field "email" is invalid',
          ],
        },
        duration: 100,
      };

      const result = responseProcessor.getErrorMessage(response);

      expect(result).toBe('Field "name" is required, Field "email" is invalid');
    });

    it('should get error message from status code and text when no error data', () => {
      const response: HttpResponse = {
        status: 500,
        statusText: 'Internal Server Error',
        headers: { 'Content-Type': 'application/json' },
        data: {},
        duration: 100,
      };

      const result = responseProcessor.getErrorMessage(response);

      expect(result).toBe('500 Internal Server Error');
    });
  });
});
