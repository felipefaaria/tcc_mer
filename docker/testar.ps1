# Atalho para testar UMA música no Music2Emo, sem digitar o comando Docker inteiro.
#
# Uso (PowerShell, a partir da raiz do projeto):
#     .\docker\testar.ps1 "data\corpus\audio\BOS01.mp3"
#     .\docker\testar.ps1 "C:\caminho\qualquer_musica.mp3"
#
# A música pode estar em QUALQUER pasta do computador. O script monta a pasta dela
# dentro do container e roda o modelo. Precisa do Docker Desktop aberto e da imagem
# 'tcc-lf1' já construída (veja docker\README.md, Passo 2, se ainda não construiu).

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Musica
)

$ErrorActionPreference = "Stop"

# Resolve o caminho e separa pasta / nome do arquivo.
try {
    $full = (Resolve-Path -LiteralPath $Musica).Path
}
catch {
    Write-Host "[erro] arquivo não encontrado: $Musica" -ForegroundColor Red
    exit 1
}
# .NET em vez de Split-Path: no PowerShell 5.1, `Split-Path -LiteralPath ... -Parent`
# dispara "conjunto de parâmetros ambíguo". GetDirectoryName/GetFileName não têm
# esse problema e lidam bem com espaços no caminho (ex.: "Área de Trabalho").
$dir  = [System.IO.Path]::GetDirectoryName($full)
$nome = [System.IO.Path]::GetFileName($full)

# Raiz do projeto = pasta acima desta (docker\). O código do pacote e dos scripts é montado
# por cima do da imagem, então mudanças nele valem sem reconstruir a imagem.
$raiz = [System.IO.Path]::GetDirectoryName($PSScriptRoot)

Write-Host ""
Write-Host ">>> Testando no Music2Emo: $nome" -ForegroundColor Cyan
Write-Host ">>> (a 1a vez baixa os pesos do modelo e demora; depois e rapido)" -ForegroundColor DarkGray

docker run --rm `
    -v "${dir}:/musica" `
    -v "${raiz}\docker:/app/docker" `
    -v "${raiz}\tcc_mer:/app/tcc_mer" `
    -v "lf1-hfcache:/root/.cache/huggingface" `
    --entrypoint python tcc-lf1 /app/docker/testar_musica.py "/musica/$nome"
