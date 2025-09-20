#!/usr/bin/env python3
"""
Script para verificar dependências antes do deploy no Railway
"""
import sys
import subprocess
import pkg_resources
from packaging import version

def check_python_version():
    """Verifica se a versão do Python é compatível"""
    print("🐍 Verificando versão do Python...")
    current_version = sys.version_info
    print(f"   Versão atual: {current_version.major}.{current_version.minor}.{current_version.micro}")
    
    if current_version.major == 3 and current_version.minor >= 11:
        print("   ✅ Versão do Python compatível")
        return True
    else:
        print("   ❌ Versão do Python incompatível (requer 3.11+)")
        return False

def check_critical_packages():
    """Verifica se os pacotes críticos podem ser instalados"""
    print("\n📦 Verificando pacotes críticos...")
    
    critical_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'lightgbm',
        'scikit-learn',
        'requests',
        'aiohttp',
        'websocket-client',
        'python-dotenv',
        'cryptography'
    ]
    
    failed_packages = []
    
    for package in critical_packages:
        try:
            pkg_resources.get_distribution(package)
            print(f"   ✅ {package}")
        except pkg_resources.DistributionNotFound:
            print(f"   ❌ {package} não encontrado")
            failed_packages.append(package)
    
    return len(failed_packages) == 0, failed_packages

def check_requirements_syntax():
    """Verifica se o requirements.txt tem sintaxe válida"""
    print("\n📄 Verificando requirements.txt...")
    
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
        
        invalid_lines = []
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#'):
                # Verificação básica de sintaxe
                if '==' not in line and '>=' not in line and '<=' not in line and '>' not in line and '<' not in line:
                    if not line.replace('-', '').replace('_', '').replace('[', '').replace(']', '').isalnum():
                        invalid_lines.append((i, line))
        
        if invalid_lines:
            print("   ❌ Linhas com sintaxe suspeita:")
            for line_num, line in invalid_lines:
                print(f"      Linha {line_num}: {line}")
            return False
        else:
            print("   ✅ Sintaxe do requirements.txt válida")
            return True
            
    except FileNotFoundError:
        print("   ❌ requirements.txt não encontrado")
        return False

def check_cryptography_compatibility():
    """Verifica especificamente a compatibilidade do cryptography"""
    print("\n🔐 Verificando compatibilidade do cryptography...")
    
    try:
        import cryptography
        crypto_version = cryptography.__version__
        print(f"   Versão instalada: {crypto_version}")
        
        # Verificar se é uma versão compatível
        if version.parse(crypto_version) >= version.parse("42.0.0"):
            print("   ✅ Versão do cryptography compatível")
            return True
        else:
            print("   ⚠️ Versão do cryptography pode ser incompatível")
            return False
            
    except ImportError:
        print("   ❌ cryptography não instalado")
        return False

def check_streamlit_config():
    """Verifica se a configuração do Streamlit está correta"""
    print("\n🎯 Verificando configuração do Streamlit...")
    
    try:
        import streamlit as st
        print("   ✅ Streamlit importado com sucesso")
        
        # Verificar se o dashboard.py existe
        import os
        if os.path.exists('dashboard.py'):
            print("   ✅ dashboard.py encontrado")
        else:
            print("   ❌ dashboard.py não encontrado")
            return False
            
        return True
        
    except ImportError as e:
        print(f"   ❌ Erro ao importar Streamlit: {e}")
        return False

def main():
    """Função principal"""
    print("🚀 Verificação de dependências para Railway Deploy")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Critical Packages", lambda: check_critical_packages()[0]),
        ("Requirements Syntax", check_requirements_syntax),
        ("Cryptography Compatibility", check_cryptography_compatibility),
        ("Streamlit Config", check_streamlit_config)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"   ❌ Erro durante verificação: {e}")
            results.append((check_name, False))
    
    print("\n" + "=" * 50)
    print("📊 RESUMO DA VERIFICAÇÃO")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"{check_name:.<30} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 TODAS AS VERIFICAÇÕES PASSARAM!")
        print("✅ Pronto para deploy no Railway!")
    else:
        print("⚠️ ALGUMAS VERIFICAÇÕES FALHARAM")
        print("❌ Corrija os problemas antes do deploy")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)