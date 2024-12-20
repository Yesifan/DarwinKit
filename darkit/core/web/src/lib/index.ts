/* eslint-disable @typescript-eslint/no-explicit-any */
// place files you want to import through the `$lib` alias in this folder.
export class ReconnectingWebSocket {
	url: string;
	protocols: string[];
	ws?: WebSocket | null;
	reconnectInterval: number;
	maxReconnectAttempts: number;
	reconnectAttempts: number;
	onMessage?: (event: any) => void;
	onOpen?: () => void;
	onClose?: (event: any) => void;
	onError?: (event: any) => void;

	constructor(url: string, protocols = []) {
		this.url = url;
		this.protocols = protocols;
		this.ws = undefined;
		this.reconnectInterval = 1000; // 重连间隔
		this.maxReconnectAttempts = 5; // 最大重连次数
		this.reconnectAttempts = 0;

		this.onMessage = undefined;
		this.onOpen = undefined;
		this.onClose = undefined;
		this.onError = undefined;

		this.connect();
	}

	connect() {
		this.ws = new WebSocket(this.url, this.protocols);

		this.ws.onopen = () => {
			this.reconnectAttempts = 0; // 重置重连次数
			if (this.onOpen) this.onOpen();
			console.log('WebSocket connection opened');
		};

		this.ws.onmessage = (event) => {
			if (this.onMessage) this.onMessage(event.data);
		};

		this.ws.onerror = (event) => {
			if (this.onError) this.onError(event);
			console.error('WebSocket error:', event);
		};

		this.ws.onclose = (event) => {
			console.log(`WebSocket closed with code: ${event.code}, reason: ${event.reason}`);

			if (event.code === 1000) {
				// 正常关闭，不再重连
				if (this.onClose) this.onClose(event);
			} else {
				// 非正常关闭，自动重连
				this.handleReconnect(event);
			}
		};
	}

	handleReconnect(event: any) {
		// 判断是否超过最大重连次数
		if (this.reconnectAttempts < this.maxReconnectAttempts) {
			this.reconnectAttempts += 1;
			console.log(
				`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`
			);
			setTimeout(() => this.connect(), this.reconnectInterval);
		} else {
			console.log('Max reconnect attempts reached. Giving up.');
			if (this.onClose) this.onClose(event);
		}
	}

	send(message: any) {
		if (this.ws && this.ws.readyState === WebSocket.OPEN) {
			this.ws.send(message);
		} else {
			console.error('WebSocket is not open. Message not sent.');
		}
	}

	close() {
		if (this.ws) {
			this.ws.close(1000, 'Client closed connection'); // 正常关闭连接
		}
	}
}
