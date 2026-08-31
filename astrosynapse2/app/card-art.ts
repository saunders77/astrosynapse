const CARD_ART_FILES = [
  "Scout.jpg",
  "Viper.jpg",
  "Explorer.jpg",
  "Battle-Blob.jpg",
  "BattlePod.jpg",
  "Blob-Carrier.jpg",
  "Blob-Destroyer.jpg",
  "Blob-Fighter.jpg",
  "Blob-Wheel.jpg",
  "Blob-World.jpg",
  "Mothership.jpg",
  "Ram.jpg",
  "The-Hive.jpg",
  "Trade-Pod.jpg",
  "Battle-Mech.jpg",
  "Battle-Station.jpg",
  "Brain-World.jpg",
  "Junkyard.jpg",
  "Machine-Base.jpg",
  "Mech-World.jpg",
  "Missile-Bot.jpg",
  "Missile-Mech.jpg",
  "Patrol-Mech.jpg",
  "Stealth-Needle.jpg",
  "Supply-Bot.jpg",
  "Trade-Bot.jpg",
  "Battlecruiser.jpg",
  "Corvette.jpg",
  "Dreadnaught.jpg",
  "Fleet-HQ.jpg",
  "Imperial-Fighter.jpg",
  "Imperial-Frigate.jpg",
  "Recycling-Station.jpg",
  "Royal-Redoubt.jpg",
  "Space-Station.jpg",
  "Survey-Ship.jpg",
  "War-World.jpg",
  "Barter-World.jpg",
  "Central-Office.jpg",
  "Command-Ship.jpg",
  "Cutter.jpg",
  "Defense-Center.jpg",
  "Embassy-Yacht.jpg",
  "Federation-Shuttle.jpg",
  "Flagship.jpg",
  "Freighter.jpg",
  "Port-of-Call.jpg",
  "Trade-Escort.jpg",
  "Trading-Post.jpg",
] as const;

const CARD_NAMES = [
  "Scout", "Viper", "Explorer", "Battle Blob", "Battle Pod", "Blob Carrier",
  "Blob Destroyer", "Blob Fighter", "Blob Wheel", "Blob World", "Mothership",
  "Ram", "The Hive", "Trade Pod", "Battle Mech", "Battle Station", "Brain World",
  "Junkyard", "Machine Base", "Mech World", "Missile Bot", "Missile Mech",
  "Patrol Mech", "Stealth Needle", "Supply Bot", "Trade Bot", "Battlecruiser",
  "Corvette", "Dreadnaught", "Fleet HQ", "Imperial Fighter", "Imperial Frigate",
  "Recycling Station", "Royal Redoubt", "Space Station", "Survey Ship", "War World",
  "Barter World", "Central Office", "Command Ship", "Cutter", "Defense Center",
  "Embassy Yacht", "Federation Shuttle", "Flagship", "Freighter", "Port of Call",
  "Trade Escort", "Trading Post",
] as const;

const normalizeCardName = (name: string) => name.trim().toLocaleLowerCase().replace(/[^a-z0-9]+/g, "");
const CARD_ID_BY_NAME = new Map(CARD_NAMES.map((name, cardId) => [normalizeCardName(name), cardId]));

export function cardArtUrlById(cardId: number | null | undefined): string | null {
  if (cardId === null || cardId === undefined || !Number.isInteger(cardId)) return null;
  const file = CARD_ART_FILES[cardId];
  return file ? `/card-art/${file}` : null;
}

export function cardArtUrlByName(name: string): string | null {
  const cardId = CARD_ID_BY_NAME.get(normalizeCardName(name));
  return cardArtUrlById(cardId);
}

export function cardArtUrl(cardId: number | null | undefined, name = ""): string | null {
  return cardArtUrlById(cardId) ?? cardArtUrlByName(name);
}
