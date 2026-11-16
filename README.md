# Fashion Garment Similarity Retrieval

## Goal

Help someone find clothing similar to an item they see in an outfit photo, without needing to describe it in a search. The project aims to turn a photo into useful suggestions from a fashion catalogue.

## Scope

The current prototype searches a fixed catalogue of 30,000 synthetic fashion products across ten categories. It handles multiple garments in one photo and returns five suggestions for each detected item, considering visual resemblance, garment category and color.

The catalogue represents clothing concepts rather than real store inventory. The goal is to suggest similar alternatives; exact product identification, shopping availability and guaranteed attribute accuracy are outside the current scope. The project uses pretrained models and focuses on building and evaluating the search experience.

## Milestones

- [x] Assemble a varied 30,000-product synthetic catalogue with product images.
- [x] Search the catalogue using a query photo.
- [x] Return separate results for multiple garments, including repeated garment types.
- [x] Use garment category and the catalogue's color labels to refine suggestions.
- [x] Handle missing detections and uncertain color predictions.
- [ ] Evaluate retrieval quality on a held-out set of real outfit photos.

## Future work

- Explore sleeve length, garment length, fit and visible patterns as additional matching signals.
- Explore a licensed real-product catalogue and larger-scale search.

## Explore the project

- [Set up the project](set-up.md)
- [Follow a sample query and view its top-five results](walkthrough.MD)
