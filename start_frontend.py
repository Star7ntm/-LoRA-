#!/usr/bin/env python3
"""
简单的HTTP服务器，用于在本地运行前端HTML文件
"""
import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8080

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def main():
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 检查frontend.html是否存在
    if not os.path.exists('frontend.html'):
        print("错误: 找不到 frontend.html 文件")
        sys.exit(1)
    
    Handler = MyHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("=" * 70)
        print("前端服务器已启动")
        print("=" * 70)
        print(f"本地访问地址: http://localhost:{PORT}/frontend.html")
        print(f"网络访问地址: http://0.0.0.0:{PORT}/frontend.html")
        print("=" * 70)
        print("提示: 确保API服务器正在运行 (http://127.0.0.1:8000)")
        print("按 Ctrl+C 停止服务器")
        print("=" * 70)
        
        # 自动打开浏览器
        try:
            webbrowser.open(f'http://localhost:{PORT}/frontend.html')
        except:
            pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")

if __name__ == "__main__":
    main()

