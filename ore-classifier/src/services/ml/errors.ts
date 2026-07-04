export class MlUnavailableError extends Error {
  constructor(message = 'Сервис анализа временно недоступен') {
    super(message);
    this.name = 'MlUnavailableError';
  }
}
