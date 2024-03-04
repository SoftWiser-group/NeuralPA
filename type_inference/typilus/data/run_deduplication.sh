# In docker image "typilus-env"
echo "In " $PWD
dotnet run --project ../tools/near-duplicate-code-detector/DuplicateCodeDetector/DuplicateCodeDetector.csproj -- --dir="./dlinfer/token/" "./dlinfer/corpus_duplicates"