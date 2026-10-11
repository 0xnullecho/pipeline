for file in data/catalogue_images/*.jpg; do
    wrangler r2 object put "threadmatch/images/$(basename "$file")" \
        --file="$file" \
        --remote
done
