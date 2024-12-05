.PHONY: example_standard
example_standard:
	mkdir -p ./output
	docker run -v ./output_images:/app/output_images tomcarter23/adversarial_attack:latest python -m adversarial_attack --model resnet50 --mode standard --image ./sample_images/imagenet/lionfish_ILSVRC2012_val_00019791.JPEG --category-truth lionfish --epsilon 1.0e-3 --max-iterations 50 --output ./output_images/adversarial_lionfish.JPEG --log DEBUG

.PHONY: example_targeted
example_targeted:
	mkdir -p ./output
	docker run -v ./output_images:/app/output_images tomcarter23/adversarial_attack:latest python -m adversarial_attack --model resnet50 --mode targeted --image ./sample_images/imagenet/lionfish_ILSVRC2012_val_00019791.JPEG --category-truth lionfish --category-target monarch --epsilon 1.0e-3 --max-iterations 50 --output ./output_images/lionfish_to_monarch.JPEG --log DEBUG
