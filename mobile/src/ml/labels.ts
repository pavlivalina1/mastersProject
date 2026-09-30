import raw from '../../assets/labels/classes_display.json';

export type ClassDisplay = {
  id: number;
  raw_label: string;
  species_en: string;
  condition_en: string;
  species_uk: string;
  condition_uk: string;
  healthy: boolean;
  care_tip_uk: string;
};

export const classes: ClassDisplay[] = raw as ClassDisplay[];

export function labelById(id: number): ClassDisplay {
  const entry = classes[id];
  if (!entry) {
    throw new Error(`No class display entry for label id ${id}`);
  }
  return entry;
}
