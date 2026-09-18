# Extension guide

Use `drosophila-repro extension-init --name <name>` outside the release asset
tree. An extension must state its scientific question, input contract, output
contract, validation criterion, and whether it is exploratory or release-grade.

Never edit frozen data or canonical renderers to test an idea. Read inputs,
write outputs under the extension directory, and add a regression test before
proposing a new release target.
