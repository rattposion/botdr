#!/usr/bin/env python3
"""
Script para corrigir problemas de codificação nos arquivos de log
"""

import os
import shutil
from datetime import datetime

def fix_log_files():
    """Corrigir arquivos de log com problemas de codificação"""
    
    log_files = [
        'logs/bot.log',
        'logs/trading_executor.log'
    ]
    
    # Criar diretório de backup
    backup_dir = f'logs/backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(backup_dir, exist_ok=True)
    
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"🔧 Processando {log_file}...")
            
            # Fazer backup
            backup_file = os.path.join(backup_dir, os.path.basename(log_file))
            shutil.copy2(log_file, backup_file)
            print(f"📁 Backup criado: {backup_file}")
            
            # Tentar ler com diferentes codificações e reescrever em UTF-8
            content_lines = []
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(log_file, 'r', encoding=encoding, errors='ignore') as f:
                        content_lines = f.readlines()
                    print(f"✅ Lido com codificação: {encoding}")
                    break
                except Exception as e:
                    print(f"❌ Erro com {encoding}: {e}")
                    continue
            
            if content_lines:
                # Reescrever em UTF-8
                try:
                    with open(log_file, 'w', encoding='utf-8') as f:
                        f.writelines(content_lines)
                    print(f"✅ {log_file} convertido para UTF-8")
                except Exception as e:
                    print(f"❌ Erro ao reescrever {log_file}: {e}")
            else:
                # Se não conseguiu ler, criar arquivo vazio
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Log reiniciado em {datetime.now().isoformat()}\n")
                print(f"🆕 {log_file} reiniciado como arquivo vazio UTF-8")
        else:
            # Criar arquivo se não existir
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"# Log criado em {datetime.now().isoformat()}\n")
            print(f"🆕 {log_file} criado como UTF-8")
    
    print(f"\n✅ Correção de logs concluída!")
    print(f"📁 Backups salvos em: {backup_dir}")

if __name__ == "__main__":
    fix_log_files()