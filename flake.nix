{
  description = "enzyme foreach comptime";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      julia = pkgs.julia-bin;
    in {
      devShells.default = pkgs.mkShell {
        packages = [julia];
      };
    });
}
